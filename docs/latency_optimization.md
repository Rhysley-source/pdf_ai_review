# Latency Optimization Guide

Routes covered: `/key-clause-extraction`, `/detect-risks`, `/red-flag-scanner`, `/generate-html`, `/regenerate-html`

---

## Current Bottlenecks (by route)

| Route | Primary Bottleneck | Secondary Bottleneck |
|---|---|---|
| `/key-clause-extraction` | 300k char context → gpt-4o-mini | 16,000 max output tokens |
| `/detect-risks` | 300k char context → gpt-4o-mini | Up to 3 LLM calls (2 attempts + fallback) |
| `/red-flag-scanner` | 2 sequential LLM calls | 80k char context in step 2 |
| `/generate-html` | 3 sequential LLM calls, step 3 uncapped tokens | No caching for repeated prompts |
| `/regenerate-html` | Extra intent-check LLM call + full generate flow | No short-circuit for obvious modifications |

---

## Suggestion 1 — Shrink Input Context (All PDF Routes)

**Files:** `feature_modules/key_clause_extraction.py:93`, `feature_modules/risk_detection.py:59`

**Problem:** Both routes set `_MAX_SINGLE_CALL_CHARS = 300_000` (~225k tokens). Most legal documents are 5–20 pages which is under 40k characters. Sending 300k chars burns token budget and increases time-to-first-token.

**Fix:** Lower the cap to 60,000 chars. This covers 30–40 dense pages — more than enough for any real-world NDA, offer letter, lease, or service agreement.

```python
# key_clause_extraction.py and risk_detection.py
_MAX_SINGLE_CALL_CHARS = 60_000  # was 300_000
```

**Impact:** ~40–60% reduction in input tokens → proportional drop in latency and cost. No accuracy loss for documents under 60k chars (the vast majority).

**When to keep 300k:** Very long contracts (100+ pages). Add a `max_chars` query param so the caller can override when needed.

---

## Suggestion 2 — Lower Output Token Caps

**File:** `feature_modules/key_clause_extraction.py:142`, `feature_modules/risk_detection.py:135`

**Problem:**
- Key clause extraction requests `max_output_tokens=16000` — a 16k token response takes 15–25s even on gpt-4o-mini.
- Risk detection requests `max_output_tokens=8000`.

Actual responses are rarely over 3,000 tokens. The model pads to the cap.

**Fix:**

```python
# key_clause_extraction.py — extract_key_clauses()
raw = await run_llm_mini(document, _SINGLE_CALL_SYSTEM, max_output_tokens=4000)  # was 16000

# risk_detection.py — analyze_document_risks()
raw = await run_llm_mini(document, _SINGLE_CALL_SYSTEM, max_output_tokens=4000)  # was 8000
```

**Impact:** 50–70% reduction in output generation time. If the model needs more tokens for a large doc, it will use them — the cap only prevents wasting time on phantom generation. Monitor logs for `finish_reason=length` and raise the cap per route if needed.

---

## Suggestion 3 — Red Flag Scanner: Pass Doc Type to Skip Step 1

**File:** `feature_modules/red_flag_scanner.py:312`, `routes/route.py:479`

**Problem:** Step 1 (`_detect_doc_type`) is a standalone LLM call (~1–2s) whose only purpose is classifying the document. If the caller already knows the doc type (e.g., the user selected it in the UI), this call is wasted.

**Fix:** Add an optional `doc_type` field to the red flag scanner endpoint. Skip detection when it is provided.

```python
# red_flag_scanner.py
async def scan_red_flags(text: str, doc_type: str | None = None) -> dict:
    if doc_type:
        checklist_key = _SLUG_TO_CHECKLIST.get(doc_type.lower(), "general")
        logger.info(f"[red_flag_scanner] doc_type provided='{doc_type}' → checklist='{checklist_key}'")
    else:
        checklist_key = await _detect_doc_type(text)
    checklist = _CHECKLISTS[checklist_key]
    # ... rest unchanged
```

```python
# route.py — red-flag-scanner endpoint
result = await scan_red_flags(text, doc_type=request.doc_type)  # pass through if provided
```

**Impact:** Saves 1–2s per request when the client sends the doc type. Zero accuracy impact — the checklist lookup is deterministic.

---

## Suggestion 4 — Red Flag Scanner: Reduce Checklist Evaluation Context

**File:** `feature_modules/red_flag_scanner.py:262`

**Problem:** Step 2 sends up to `text[:80_000]` (80k chars ≈ 60k tokens) plus the checklist prompt. For most documents, the checklist items (12–18 questions) can be answered from the first 20–30 pages.

**Fix:** Reduce context to 30,000 chars for step 2:

```python
# red_flag_scanner.py — _build_eval_messages()
{text[:30_000]}   # was 80_000
```

**Impact:** ~50% fewer input tokens in step 2, saving 2–4s. For very long documents, the most critical red-flag clauses always appear in the first 20–30 pages.

---

## Suggestion 5 — In-Memory Result Cache (All Routes)

**File:** New utility, used in `routes/route.py`

**Problem:** No caching exists. The same PDF uploaded twice triggers a full LLM pipeline both times.

**Fix:** Add a simple TTL cache keyed on `SHA-256(text + endpoint)`:

```python
# utils/cache.py
import hashlib
import time

_cache: dict[str, tuple[dict, float]] = {}
_TTL_SECONDS = 3600  # 1 hour

def cache_key(text: str, endpoint: str) -> str:
    return hashlib.sha256(f"{endpoint}:{text[:5000]}".encode()).hexdigest()

def get_cached(key: str) -> dict | None:
    entry = _cache.get(key)
    if entry and (time.time() - entry[1]) < _TTL_SECONDS:
        return entry[0]
    _cache.pop(key, None)
    return None

def set_cached(key: str, value: dict) -> None:
    _cache[key] = (value, time.time())
```

```python
# route.py — inside each endpoint handler
from utils.cache import cache_key, get_cached, set_cached

key    = cache_key(text, "/key-clause-extraction")
cached = get_cached(key)
if cached:
    return cached

result = await extract_key_clauses(text)
set_cached(key, result)
return result
```

**Impact:** Zero latency on cache hit (repeat requests). Works especially well for `/generate-html` where the same prompt is sent multiple times during testing.

**Note:** Use Redis instead of `_cache` dict for multi-worker deployments (Gunicorn with multiple processes).

---

## Suggestion 6 — Generate HTML: Parallelize Steps 1 and 2

**File:** `document_generation/document_generator.py`

**Problem:** The generate flow runs Step 1 (query analysis) → Step 2 (blueprint) → Step 3 (HTML) sequentially. Steps 1 and 2 both use the fast model (`gpt-4.1-nano`), but Step 2 depends on Step 1's output — so they truly must be sequential.

However, any **pre-computation that doesn't depend on step 1** can start immediately. For `/regenerate-html`, the intent check (`_check_regeneration_intent`) and the document load (`_load_document`) currently run sequentially but are independent.

**Fix for `/regenerate-html`:**

```python
# document_generator.py — regenerate_document_html()
# Run intent check and doc load in parallel
existing_html, intent = await asyncio.gather(
    asyncio.to_thread(_load_document, request.document_id),
    _check_regeneration_intent(request.modification_query, stored_html_placeholder)
)
```

**Note:** Intent check needs the HTML content, so load the document first, then run both intent check and any other pre-work in parallel.

**Impact:** Saves 0.5–1.5s on `/regenerate-html` by overlapping the intent check with document loading I/O.

---

## Suggestion 7 — Generate HTML: Stream Step 3 to Client

**File:** `document_generation/document_generator.py:756`

**Problem:** `/generate-html` and `/regenerate-html` wait for the full HTML to be generated (Step 3) before sending anything to the client. Step 3 is uncapped and can take 10–20s for complex documents.

**Fix:** Add a `StreamingResponse` variant using the OpenAI streaming API for Step 3. The client receives HTML tokens as they arrive, making it feel instant even for long documents.

```python
@router.post("/generate-html/stream")
async def generate_document_html_stream(request: DocumentGenerationRequest):
    # Steps 1 + 2 run normally (fast)
    context = await _build_context(request.user_prompt)

    async def streamer():
        system_prompt = DOCUMENT_GENERATION_V2_PROMPT.format(**context, user_request=request.user_prompt)
        async with await _CLIENT.chat.completions.create(
            model=_MODEL, stream=True,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user",   "content": request.user_prompt}],
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    return StreamingResponse(streamer(), media_type="text/html")
```

**Impact:** Perceived latency drops to near-zero (user sees content streaming immediately). Actual generation time is unchanged but the user experience is dramatically better.

---

## Suggestion 8 — PDF OCR: Skip OCR for Text-Based PDFs

**File:** `utils/pdf_utils.py` (called by `feature_modules/key_clause_extraction.py:283`)

**Problem:** `load_pdf` currently runs the same extraction path for all PDFs. Most modern PDFs (contracts, offer letters) have embedded text — they don't need OCR. OCR is only needed for scanned/image PDFs and adds 2–5s per page.

**Fix:** Try direct text extraction first. Fall back to OCR only if the extracted text is empty or below a character threshold.

```python
# utils/pdf_utils.py
def load_pdf(file_path: str, max_pages: int) -> list:
    pages = _extract_embedded_text(file_path, max_pages)
    if _has_enough_text(pages):
        return pages          # fast path — no OCR needed
    return _extract_with_ocr(file_path, max_pages)   # slow path

def _has_enough_text(pages: list, min_chars_per_page: int = 100) -> bool:
    return all(len(p.page_content.strip()) >= min_chars_per_page for p in pages)
```

**Impact:** 2–5s saved per page for text-based PDFs (which is most documents). OCR path is unchanged for scanned images.

---

## Priority Order

| Priority | Suggestion | Effort | Latency Saving | Accuracy Risk |
|---|---|---|---|---|
| 1 | Shrink input context to 60k chars | Low | 40–60% on LLM input | None |
| 2 | Lower output token caps | Low | 50–70% on output generation | None (monitor logs) |
| 3 | Pass doc_type to skip red-flag detection | Low | 1–2s on red-flag scanner | None |
| 4 | In-memory result cache | Medium | 100% on cache hit | None |
| 5 | Skip OCR for text-based PDFs | Medium | 2–5s per page | None |
| 6 | Red-flag scanner: shrink step 2 context | Low | 2–4s | Minimal |
| 7 | Parallelize regenerate-html pre-steps | Medium | 0.5–1.5s | None |
| 8 | Stream generate-html step 3 | High | Perceived only | None |

---

## Expected Combined Impact

Applying suggestions 1–6 together:

| Route | Current (est.) | After Optimization | Saving |
|---|---|---|---|
| `/key-clause-extraction` | 12–20s | 4–7s | ~60% |
| `/detect-risks` | 10–18s | 3–6s | ~65% |
| `/red-flag-scanner` | 8–14s | 3–6s | ~55% |
| `/generate-html` | 15–30s | 10–20s (8–12s perceived with streaming) | ~40% |
| `/regenerate-html` | 17–33s | 11–22s | ~35% |
