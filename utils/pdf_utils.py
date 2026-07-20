import re
import os
import time
import logging
import httpx
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking config
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 12000
CHUNK_OVERLAP = 200

# Pre-compiled regex patterns
_RE_HYPHEN    = re.compile(r"-\n")
_RE_MULTILINE = re.compile(r"\n{3,}")
_RE_SPACES    = re.compile(r" {2,}")

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
NATIVE_TEXT_THRESHOLD = 0

# Parallel workers for native extraction (fitz is thread-safe for reads)
_NATIVE_EXTRACT_WORKERS = 4

# How many pages to sample when detecting PDF type
_PDF_TYPE_SAMPLE_PAGES = 5

# ---------------------------------------------------------------------------
# Remote OCR API — used for scanned/image pages that PyMuPDF can't read
# natively. Replaces the previous local PaddleOCR-VL GPU pipeline.
#
# .env:
#   OCR_API_URL   = https://raceai.studyineurope.xyz/v1/ocr
#   OCR_API_KEY   = sk_live_...
#   OCR_API_PROMPT (optional, default "")
# ---------------------------------------------------------------------------
OCR_API_URL    = os.environ.get("OCR_API_URL", "https://raceai.studyineurope.xyz/v1/ocr")
OCR_API_KEY    = os.environ.get("OCR_API_KEY", "")
OCR_API_PROMPT = os.environ.get("OCR_API_PROMPT", "")

# Timeout scales with page count — the API OCRs the whole sub-PDF in one call.
OCR_API_TIMEOUT_BASE_S      = int(os.environ.get("OCR_API_TIMEOUT_BASE_S", "60"))
OCR_API_TIMEOUT_PER_PAGE_S  = int(os.environ.get("OCR_API_TIMEOUT_PER_PAGE_S", "20"))

if not OCR_API_KEY:
    logger.warning(
        "[pdf_utils] OCR_API_KEY is not set — scanned/image PDF pages will "
        "fail extraction until it is added to the environment."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean PDF/OCR artifacts. Never returns None."""
    before = len(text)
    text = _RE_HYPHEN.sub("",        text)
    text = _RE_MULTILINE.sub("\n\n", text)
    text = _RE_SPACES.sub(" ",       text)
    text = text.strip()
    logger.debug(f"[pdf_utils] clean_text: {before} -> {len(text)} chars")
    return text


def _detect_pdf_type(doc: fitz.Document, pages_to_process: int) -> str:
    """
    Sample the first N pages to classify the PDF before extraction begins.

    Returns:
      'native'     - all sampled pages have text  -> text-based PDF
      'image_only' - NO sampled pages have text   -> fully scanned/image PDF
      'mixed'      - some have text, some do not  -> hybrid PDF
    """
    sample       = min(_PDF_TYPE_SAMPLE_PAGES, pages_to_process)
    native_count = 0

    for i in range(sample):
        text = doc[i].get_text("text").strip()
        if len(text) > 0:
            native_count += 1

    if native_count == sample:
        pdf_type = "native"
    elif native_count == 0:
        pdf_type = "image_only"
    else:
        pdf_type = "mixed"

    logger.info(
        f"[pdf_utils] PDF type: sampled {sample} page(s), "
        f"{native_count} had native text -> '{pdf_type}'"
    )
    return pdf_type


def _extract_native(page: fitz.Page) -> str | None:
    """Strategy 1 - PyMuPDF native. Returns text if any chars present, else None."""
    text = page.get_text("text").strip()
    if len(text) > NATIVE_TEXT_THRESHOLD:
        return clean_text(text)
    return None


# ---------------------------------------------------------------------------
# Blank-PDF detection
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(
    r"^\[Page \d+: (blank page|content could not be extracted)\]$"
)


def all_pages_blank(pages: list[Document]) -> bool:
    """
    Return True if every page in the list is a blank/placeholder page.
    """
    if not pages:
        return True
    for page in pages:
        text = page.page_content.strip()
        if text and not _PLACEHOLDER_RE.match(text):
            return False
    return True


# ---------------------------------------------------------------------------
# Remote OCR call — sends a (sub-)PDF of the pages that need OCR to the
# external OCR API and returns page texts in the order the API returned them.
# ---------------------------------------------------------------------------

def _call_remote_ocr(pdf_bytes: bytes, num_pages: int) -> list[str]:
    """
    POST a PDF to the remote OCR API and return one text string per page,
    in page order. Raises on network/HTTP failure or a malformed response —
    caller is responsible for turning that into placeholder pages.
    """
    if not OCR_API_KEY:
        raise RuntimeError("OCR_API_KEY is not configured.")

    timeout_s = OCR_API_TIMEOUT_BASE_S + OCR_API_TIMEOUT_PER_PAGE_S * num_pages

    response = httpx.post(
        OCR_API_URL,
        headers={
            "accept":    "application/json",
            "x-api-key": OCR_API_KEY,
        },
        files={"file": ("document.pdf", pdf_bytes, "application/pdf")},
        data={"prompt": OCR_API_PROMPT},
        timeout=timeout_s,
    )
    response.raise_for_status()
    body = response.json()
    logger.info(f"[pdf_utils] Remote OCR API response: {body}")

    if not body.get("success"):
        raise RuntimeError(f"Remote OCR API reported failure: {body}")

    pages_out = sorted(body.get("pages", []), key=lambda p: p.get("page", 0))
    return [(p.get("text") or "").strip() for p in pages_out]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pdf(file_path: str, max_pages: int | None = None, _stats: dict | None = None) -> list[Document]:
    """
    Load a PDF and return one LangChain Document per page.

    Extraction pipeline — checked per page individually:

      Tier 1 — PyMuPDF native   (parallel, ~0.01s/page, no network cost)
                If a page yields text → done.
      Tier 2 — Remote OCR API  (all pages needing OCR sent in one batched
                call, see _call_remote_ocr) — any page whose PyMuPDF
                result is empty (image/scanned) → OCR.
      Tier 3 — Placeholder     ("[Page N: content could not be extracted]")
                Used when the remote OCR call fails.

    OCR timeout scales with page count (OCR_API_TIMEOUT_BASE_S +
    OCR_API_TIMEOUT_PER_PAGE_S * pages). If the call fails or times out,
    every page in that batch gets a placeholder rather than blocking the
    server indefinitely.

    pdf_type ('native' / 'image_only' / 'mixed') is detected for stats/logging
    only — it does NOT skip any extraction pass.

    Data integrity: every page is always returned. No page is silently dropped.
    """
    logger.info(
        f"[pdf_utils] load_pdf: '{file_path}'"
        + (f" max_pages={max_pages}" if max_pages else "")
    )
    t_load = time.perf_counter()

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"[pdf_utils] load_pdf: open failed -- {e}")
        raise ValueError(f"Could not open PDF: {e}")

    total_pages      = len(doc)
    pages_to_process = total_pages if max_pages is None else min(total_pages, max_pages)
    logger.info(
        f"[pdf_utils] {total_pages} total page(s), "
        f"processing {pages_to_process} "
        f"({'all' if pages_to_process == total_pages else f'first {pages_to_process}'})"
    )

    pdf_type   = _detect_pdf_type(doc, pages_to_process)   # stats/logging only
    fitz_pages = [doc[i] for i in range(pages_to_process)]

    # ── Pass 1: PyMuPDF on every page (parallel) ─────────────────────────
    # Pages with no native text are sent to OCR regardless of pdf_type.
    native_results: dict[int, str | None] = {}

    logger.info(
        f"[pdf_utils] Pass 1 -- PyMuPDF on all {pages_to_process} page(s) "
        f"({_NATIVE_EXTRACT_WORKERS} workers)"
    )
    t_pass1 = time.perf_counter()

    def _native_worker(idx_page):
        idx, page = idx_page
        return idx, _extract_native(page)

    with ThreadPoolExecutor(max_workers=_NATIVE_EXTRACT_WORKERS) as pool:
        futures = {
            pool.submit(_native_worker, (i, p)): i
            for i, p in enumerate(fitz_pages)
        }
        for future in as_completed(futures):
            try:
                idx, text = future.result()
                native_results[idx] = text
            except Exception as e:
                logger.warning(f"[pdf_utils] Pass 1 worker failed: {e}")
                native_results[futures[future]] = None

    native_hit       = sum(1 for v in native_results.values() if v is not None)
    _t_pass1_elapsed = time.perf_counter() - t_pass1
    logger.info(
        f"[pdf_utils] Pass 1 done ({_t_pass1_elapsed:.2f}s) -- "
        f"native={native_hit}, need_ocr={pages_to_process - native_hit}"
    )
    if _stats is not None:
        _stats["pymupdf_time"] = _t_pass1_elapsed
        _stats["native_pages"] = native_hit

    ocr_needed = sorted(i for i, v in native_results.items() if v is None)

    # ── Pass 2: remote OCR API for scanned/image pages ─────────────────────
    # All pages needing OCR are collected into a single sub-PDF and sent to
    # the remote OCR API in one call (cheaper and faster than one call/page).
    ocr_results:        dict[int, str] = {}
    _t_ocr_total       = 0.0
    _ocr_timeout_count = 0

    if ocr_needed:
        logger.info(
            f"[pdf_utils] Pass 2 -- remote OCR API for {len(ocr_needed)} page(s)"
        )
        t_ocr = time.perf_counter()

        sub_doc = fitz.open()
        try:
            for idx in ocr_needed:
                sub_doc.insert_pdf(doc, from_page=idx, to_page=idx)
            pdf_bytes = sub_doc.tobytes()
        finally:
            sub_doc.close()

        try:
            texts   = _call_remote_ocr(pdf_bytes, len(ocr_needed))
            elapsed = time.perf_counter() - t_ocr
            _t_ocr_total += elapsed

            if len(texts) != len(ocr_needed):
                logger.warning(
                    f"[pdf_utils] Remote OCR returned {len(texts)} page(s), "
                    f"expected {len(ocr_needed)} — mapping by position"
                )

            for pos, idx in enumerate(ocr_needed):
                page_num = idx + 1
                text     = clean_text(texts[pos]) if pos < len(texts) else ""
                if not text:
                    logger.info(f"[pdf_utils] Page {page_num}: remote OCR returned no text — blank page")
                    ocr_results[idx] = f"[Page {page_num}: blank page]"
                else:
                    ocr_results[idx] = text

            logger.info(
                f"[pdf_utils] Remote OCR done ({elapsed:.2f}s) — "
                f"{len(ocr_needed)} page(s) processed"
            )

        except Exception as e:
            elapsed = time.perf_counter() - t_ocr
            _t_ocr_total += elapsed
            logger.error(
                f"[pdf_utils] Remote OCR call failed for {len(ocr_needed)} page(s) "
                f"({elapsed:.2f}s) — {e}"
            )
            _ocr_timeout_count += len(ocr_needed)
            for idx in ocr_needed:
                ocr_results[idx] = f"[Page {idx + 1}: content could not be extracted]"

    # ── Assemble results in page order ────────────────────────────────────
    pages:             list[Document] = []
    ocr_count:         int = 0
    placeholder_count: int = 0
    blank_count:       int = 0

    for idx in range(pages_to_process):
        page_num = idx + 1

        if native_results.get(idx) is not None:
            text = native_results[idx]
            logger.debug(f"[pdf_utils] Page {page_num}: native ({len(text)} chars)")

        elif idx in ocr_needed:
            text = ocr_results.get(idx, f"[Page {page_num}: content could not be extracted]")
            if text.startswith("[Page "):
                placeholder_count += 1
            else:
                ocr_count += 1

        else:
            logger.error(f"[pdf_utils] Page {page_num}: no result -- placeholder")
            text = f"[Page {page_num}: content could not be extracted]"
            placeholder_count += 1

        if not text:
            logger.warning(f"[pdf_utils] Page {page_num}: blank after extraction")
            text = f"[Page {page_num}: blank page]"
            blank_count += 1

        pages.append(Document(
            page_content=text,
            metadata={"page": page_num, "source": file_path},
        ))

    doc.close()
    elapsed      = time.perf_counter() - t_load
    total_loaded = len(pages)

    if _stats is not None:
        _stats.setdefault("pymupdf_time", 0.0)
        _stats.setdefault("native_pages", native_hit)
        _stats["ocr_time"]          = _t_ocr_total
        _stats["ocr_pages"]         = ocr_count
        _stats["total_time"]        = elapsed
        _stats["pdf_type"]          = pdf_type
        _stats["placeholder_pages"] = placeholder_count
        _stats["blank_pages"]       = blank_count
        _stats["ocr_timeout_pages"] = _ocr_timeout_count

    # ── Integrity report ──────────────────────────────────────────────────
    logger.info("[pdf_utils] -- EXTRACTION COMPLETE --------------------------")
    logger.info(f"[pdf_utils] PDF type         : {pdf_type}")
    logger.info(f"[pdf_utils] Pages processed  : {pages_to_process}/{total_pages}")
    logger.info(f"[pdf_utils] Native text       : {native_hit} page(s)")
    logger.info(f"[pdf_utils] Remote OCR        : {ocr_count} page(s)")
    logger.info(f"[pdf_utils] OCR timeouts       : {_ocr_timeout_count} page(s)")
    logger.info(f"[pdf_utils] Placeholders      : {placeholder_count} page(s)")
    logger.info(f"[pdf_utils] Blank pages        : {blank_count} page(s)")
    logger.info(f"[pdf_utils] Total loaded      : {total_loaded} page(s)")
    logger.info(f"[pdf_utils] Time              : {elapsed:.2f}s")
    logger.info("[pdf_utils] ------------------------------------------------------")

    if placeholder_count > 0:
        logger.warning(
            f"[pdf_utils] {placeholder_count} page(s) could not be extracted -- "
            "included as placeholders"
        )

    if total_loaded != pages_to_process:
        logger.error(
            f"[pdf_utils] DATA INTEGRITY: expected {pages_to_process}, got {total_loaded}"
        )

    if total_loaded == 0:
        raise ValueError(
            "No content extracted from any page. "
            "File may be blank, encrypted, or corrupt."
        )

    return pages


def merge_pages(pages: list[Document]) -> list[Document]:
    """Concatenate all pages into one Document so the splitter produces full-size chunks."""
    if not pages:
        return pages

    total_chars_before = sum(len(p.page_content) for p in pages)
    logger.info(
        f"[pdf_utils] merge_pages: {len(pages)} page(s), "
        f"{total_chars_before:,} total chars"
    )

    combined_text = "\n\n".join(p.page_content for p in pages)

    if len(combined_text) < total_chars_before:
        logger.error(
            f"[pdf_utils] merge_pages: DATA LOSS -- "
            f"merged ({len(combined_text):,}) < sum of pages ({total_chars_before:,})"
        )

    merged = Document(
        page_content=combined_text,
        metadata={
            "page":   f"1-{pages[-1].metadata.get('page', len(pages))}",
            "source": pages[0].metadata.get("source", ""),
        },
    )
    logger.info(
        f"[pdf_utils] merge_pages: {len(combined_text):,} chars "
        f"(~{len(combined_text) // CHUNK_SIZE + 1} chunks)"
    )
    return [merged]


def split_documents(pages: list[Document]) -> list[Document]:
    """Merge all pages then split into CHUNK_SIZE chunks."""
    merged      = merge_pages(pages)
    total_chars = sum(len(p.page_content) for p in merged)

    logger.info(
        f"[pdf_utils] split_documents: {total_chars:,} chars, "
        f"chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    t_split = time.perf_counter()
    chunks  = splitter.split_documents(merged)
    avg     = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0

    logger.info(
        f"[pdf_utils] split_documents: {len(chunks)} chunk(s) in "
        f"{time.perf_counter() - t_split:.3f}s (avg {avg:.0f} chars/chunk)"
    )
    logger.info(
        f"[pdf_utils] Coverage: ALL {len(pages)} page(s) -> "
        f"{len(chunks)} inference call(s)"
    )
    return chunks


def get_page_count(file_path: str) -> int:
    """Return page count without fully loading the PDF."""
    try:
        with fitz.open(file_path) as doc:
            count = len(doc)
            logger.debug(f"[pdf_utils] get_page_count: {count} page(s)")
            return count
    except Exception as e:
        logger.error(f"[pdf_utils] get_page_count failed: {e}")
        return 0