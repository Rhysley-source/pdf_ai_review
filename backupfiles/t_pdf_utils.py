"""
t_pdf_utils.py
==============
PDF text extraction — supports small PDFs, large PDFs (500+ pages),
text-based PDFs, image-only / scanned PDFs, and mixed PDFs.

Features
--------
- Parallel batch page extraction  (BATCH_SIZE pages concurrently)
- No hard page limit              (handles 500+ pages)
- Smart page classification       (native text vs OCR vs blank)
- Adaptive chunk sizing           (dense / normal / sparse content)
- Grayscale + RGBA → RGB          (all pixmap formats handled)
- Safe PaddleOCR result unpacking (str or dict res.res)
- GPU memory freed after every OCR call
- Clear RuntimeError for image-only PDFs when OCR is unavailable
- Native text preserved as fallback when OCR returns nothing

Edge cases handled
------------------
A) Image-only PDF with OCR unavailable
   → RuntimeError with actionable install message (caught by t_main.py)

B) Pages with < 50 native chars  (old hard threshold removed)
   → _classify_page() keeps short-but-real text (title pages, headers)
   → Sends noisy/symbol pages to OCR with native text as fallback
"""

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor

import fitz          # PyMuPDF
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BATCH_SIZE       = 20      # pages extracted concurrently per batch
OCR_DPI          = 150     # render resolution for scanned pages
MIN_NATIVE_CHARS = 10      # native chars below this → always try OCR
WORD_RATIO_MIN   = 0.40    # min fraction of tokens that look like real words

# Adaptive chunk sizing (characters)
CHUNK_DENSE      = 5_000   # dense legal / technical prose
CHUNK_NORMAL     = 10_000  # standard mixed content
CHUNK_SPARSE     = 15_000  # tables, indices, appendices

# Shared thread pool — one per process, sized to batch width
_PAGE_EXECUTOR   = ThreadPoolExecutor(max_workers=BATCH_SIZE)

# ---------------------------------------------------------------------------
# Optional PaddleOCR-VL  (graceful degradation if not installed)
# ---------------------------------------------------------------------------
try:
    import paddle
    from paddleocr import PaddleOCRVL
    _ocr_vl           = PaddleOCRVL("v1.5")
    _PADDLE_AVAILABLE  = True
    logger.info("PaddleOCR-VL loaded successfully.")
except Exception as _exc:
    logger.warning(f"PaddleOCR-VL not available ({_exc}). OCR disabled.")
    _ocr_vl           = None
    _PADDLE_AVAILABLE  = False


# ---------------------------------------------------------------------------
# Page classification
# ---------------------------------------------------------------------------

def _classify_page(native: str) -> str:
    """
    Decide how to process a page based on its native PyMuPDF text.

    Returns
    -------
    "native"     — enough real text; skip OCR entirely
    "ocr_needed" — insufficient or noisy native text; attempt OCR
                   (native text kept as fallback if OCR returns "")
    "blank"      — completely empty; attempt OCR (may be image page)

    Rules
    -----
    - "" or None                           → "blank"
    - < MIN_NATIVE_CHARS chars             → "ocr_needed"
    - >= MIN_NATIVE_CHARS, high word ratio → "native"
    - >= MIN_NATIVE_CHARS, low word ratio  → "ocr_needed"  (encoding noise)
    """
    stripped = (native or "").strip()

    if not stripped:
        return "blank"

    char_count = len(stripped)

    if char_count < MIN_NATIVE_CHARS:
        return "ocr_needed"

    tokens     = stripped.split()
    real_words = sum(1 for t in tokens if sum(c.isalpha() for c in t) >= 2)
    ratio      = real_words / max(len(tokens), 1)

    if ratio >= WORD_RATIO_MIN:
        return "native"

    return "ocr_needed"


# ---------------------------------------------------------------------------
# Low-level helpers (synchronous — run inside thread pool)
# ---------------------------------------------------------------------------

def _page_to_rgb(pix: fitz.Pixmap) -> np.ndarray:
    """Convert a PyMuPDF Pixmap to an HxWx3 uint8 RGB numpy array.
    Handles n=1 (grayscale), n=3 (RGB), and n=4 (RGBA)."""
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 1:       # grayscale → RGB
        img = np.repeat(img, 3, axis=2)
    elif pix.n == 4:     # RGBA → RGB
        img = img[:, :, :3]
    return img


def _run_ocr_sync(img: np.ndarray) -> str:
    """
    Run PaddleOCR-VL on a single RGB page image.

    Safely unpacks res.res whether it is a plain string or a dict
    (PaddleOCR-VL returns dicts like {"rec_text": "...", ...}).
    Always frees GPU memory before returning.
    """
    if _ocr_vl is None:
        return ""

    results = None
    try:
        results    = _ocr_vl.predict(img)
        page_parts = []

        for res in results:
            # Unpack result object
            if hasattr(res, "res"):
                raw = res.res
            elif isinstance(res, dict) and "res" in res:
                raw = res["res"]
            else:
                raw = str(res)

            # res.res can be a dict → extract the text field
            if isinstance(raw, dict):
                text = (
                    raw.get("rec_text")
                    or raw.get("text")
                    or " ".join(str(v) for v in raw.values() if isinstance(v, str))
                )
            else:
                text = str(raw)

            if text and text.strip():
                page_parts.append(text.strip())

        return "\n".join(page_parts)

    except Exception as exc:
        logger.error(f"OCR predict error: {exc}")
        return ""

    finally:
        # Always free GPU memory
        try:
            del results
        except Exception:
            pass
        if _PADDLE_AVAILABLE:
            try:
                if paddle.device.is_compiled_with_cuda():
                    paddle.device.cuda.empty_cache()
            except Exception:
                pass


def _extract_page_sync(file_path: str, page_num: int) -> tuple[int, str]:
    """
    Extract text from ONE page. Runs inside a thread-pool worker.

    Each call opens its own fitz.Document handle so threads never
    share document state. The pixmap is deleted immediately after OCR
    to prevent RAM accumulation across hundreds of pages.

    Returns (page_num, text) so results can be sorted back into order.

    Strategy
    --------
    1. Native text extraction via PyMuPDF.
    2. _classify_page() decides whether native text is sufficient.
    3. If OCR is needed and available → OCR; keep native as fallback.
    4. If OCR is needed but unavailable → return whatever native text exists.
    """
    try:
        with fitz.open(file_path) as doc:
            page   = doc[page_num]
            native = page.get_text("text") or ""
            kind   = _classify_page(native)

            # ── Case 1: native text is good ──────────────────────────────────
            if kind == "native":
                logger.debug(f"Page {page_num+1}: native ({len(native.strip())} chars)")
                return page_num, native.strip()

            # ── Case 2 & 3: OCR needed ───────────────────────────────────────
            if not _PADDLE_AVAILABLE:
                fallback = native.strip()
                if fallback:
                    logger.warning(
                        f"Page {page_num+1}: OCR unavailable, "
                        f"keeping {len(fallback)}-char native text as fallback."
                    )
                else:
                    logger.warning(
                        f"Page {page_num+1}: image page and OCR unavailable — skipped."
                    )
                return page_num, fallback

            # Run OCR
            try:
                pix      = page.get_pixmap(dpi=OCR_DPI)
                img      = _page_to_rgb(pix)
                del pix                          # free pixmap RAM immediately
                ocr_text = _run_ocr_sync(img)

                if ocr_text:
                    logger.debug(
                        f"Page {page_num+1}: OCR → {len(ocr_text)} chars"
                    )
                    return page_num, ocr_text

                # OCR returned nothing → fall back to native text (never discard)
                fallback = native.strip()
                if fallback:
                    logger.warning(
                        f"Page {page_num+1}: OCR returned empty; "
                        f"keeping {len(fallback)}-char native fallback."
                    )
                else:
                    logger.warning(f"Page {page_num+1}: OCR empty, no native text.")
                return page_num, fallback

            except MemoryError:
                logger.error(
                    f"Page {page_num+1}: MemoryError during OCR — using native fallback."
                )
                return page_num, native.strip()
            except Exception as exc:
                logger.error(
                    f"Page {page_num+1}: OCR error ({exc}) — using native fallback."
                )
                return page_num, native.strip()

    except Exception as exc:
        logger.error(f"Page {page_num+1}: could not open document ({exc}).")
        return page_num, ""


# ---------------------------------------------------------------------------
# Public extraction API
# ---------------------------------------------------------------------------

async def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF of any size using async parallel page batches.

    Algorithm
    ---------
    1. Open the PDF once to count pages, then close it.
    2. Divide all page numbers into batches of BATCH_SIZE.
    3. For each batch, run every page in _PAGE_EXECUTOR concurrently
       via asyncio.gather (non-blocking for the event loop).
    4. Sort (page_num, text) results, join, and clean.

    Edge cases
    ----------
    - All pages empty + OCR unavailable  → RuntimeError (actionable message)
    - All pages empty + OCR available    → RuntimeError (blank/corrupt PDF)
    - Some pages empty                   → warning logged, others used

    Raises
    ------
    ValueError   — file cannot be opened (corrupt / encrypted)
    RuntimeError — no text could be extracted from any page
    """
    try:
        with fitz.open(file_path) as probe:
            total_pages = len(probe)
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}")

    logger.info(
        f"Extracting {total_pages} pages — "
        f"{-(-total_pages // BATCH_SIZE)} batch(es) of {BATCH_SIZE}"
    )

    batches = [
        list(range(i, min(i + BATCH_SIZE, total_pages)))
        for i in range(0, total_pages, BATCH_SIZE)
    ]

    loop         = asyncio.get_running_loop()
    page_results: list[tuple[int, str]] = []

    for b_idx, batch in enumerate(batches):
        logger.info(
            f"Batch {b_idx+1}/{len(batches)}: "
            f"pages {batch[0]+1}–{batch[-1]+1}"
        )
        futures    = [
            loop.run_in_executor(_PAGE_EXECUTOR, _extract_page_sync, file_path, pn)
            for pn in batch
        ]
        batch_out  = await asyncio.gather(*futures)
        page_results.extend(batch_out)

    page_results.sort(key=lambda t: t[0])

    # ── Edge case A: detect image-only PDF ──────────────────────────────────
    empty_count = sum(1 for _, t in page_results if not t.strip())

    if empty_count == total_pages:
        if not _PADDLE_AVAILABLE:
            raise RuntimeError(
                "This PDF is entirely image-based (scanned or photo PDF) and "
                "contains no extractable native text. "
                "PaddleOCR-VL is not installed, so OCR cannot be performed. "
                "Install it with:  pip install paddlepaddle paddleocr  "
                "then restart the server."
            )
        else:
            raise RuntimeError(
                "No text could be extracted from this PDF even with OCR. "
                "The file may be blank, password-protected, or corrupt."
            )

    if empty_count > 0:
        logger.warning(
            f"{empty_count}/{total_pages} pages yielded no text "
            f"(image pages skipped or blank pages)."
        )

    joined = "\n".join(text for _, text in page_results if text.strip())
    return clean_text(joined)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean common PDF extraction artefacts:
      - Rejoin hyphenated line-breaks  (word-\nbreak → wordbreak)
      - Collapse 3+ blank lines to double newline
      - Collapse repeated spaces
    """
    text = re.sub(r"-\n",    "",      text)
    text = re.sub(r"\n{3,}", "\n\n",  text)
    text = re.sub(r" {2,}",  " ",     text)
    return text.strip()


def adaptive_chunk_size(text: str) -> int:
    """
    Choose a chunk character limit based on word density.

    Density  > 0.15 words/char  →  CHUNK_DENSE  (5 000)  — legal / technical prose
    Density  > 0.08 words/char  →  CHUNK_NORMAL (10 000) — standard mixed content
    Density ≤ 0.08 words/char  →  CHUNK_SPARSE (15 000) — tables / indices
    """
    if not text:
        return CHUNK_NORMAL
    density = len(text.split()) / max(len(text), 1)
    if density > 0.15:
        return CHUNK_DENSE
    elif density > 0.08:
        return CHUNK_NORMAL
    return CHUNK_SPARSE


def chunk_text(text: str, max_chars: int = 0) -> list[str]:
    """
    Split text into chunks on paragraph boundaries (never mid-sentence).

    Parameters
    ----------
    text      : cleaned extracted text
    max_chars : 0 = choose automatically via adaptive_chunk_size()
                >0 = use this fixed size

    Returns
    -------
    List of non-empty string chunks.
    """
    if not text:
        return []

    size       = max_chars if max_chars > 0 else adaptive_chunk_size(text)
    paragraphs = text.split("\n")
    chunks: list[str] = []
    current    = ""

    for para in paragraphs:
        if len(current) + len(para) > size:
            if current.strip():
                chunks.append(current.strip())
            current = para
        else:
            current += "\n" + para

    if current.strip():
        chunks.append(current.strip())

    logger.info(
        f"chunk_text: {len(text):,} chars → {len(chunks)} chunks "
        f"(target {size:,} chars each)"
    )
    return chunks