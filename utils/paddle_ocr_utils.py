"""
paddle_ocr_utils.py

Standalone local PaddleOCR-VL 1.5 pipeline, exposed as its own endpoint
(POST /ocr/paddle-vl in routes/route.py) rather than being wired into the
main load_pdf() extraction pipeline.

load_pdf() (utils/pdf_utils.py) now uses a remote OCR API for scanned pages —
this module keeps the previous local GPU PaddleOCR-VL path available for
direct testing/comparison without affecting the main /analyze pipeline.

The model is loaded lazily on first call, not at import time — routes/route.py
is imported once at app startup, so eagerly loading PaddleOCR-VL here would
slow every worker's startup and load it even in workers that never receive
an OCR request (same reasoning as model_file.py's Phi model).

.env (optional):
  PADDLE_OCR_PAGE_TIMEOUT  default: 30   (seconds per page)
  PADDLE_OCR_DPI           default: 150  (page render resolution)
"""

import os
import time
import logging
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import numpy as np
import fitz  # PyMuPDF

from utils.pdf_utils import clean_text

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor",
    category=UserWarning,
)

OCR_PAGE_TIMEOUT = int(os.environ.get("PADDLE_OCR_PAGE_TIMEOUT", "30"))
OCR_DPI          = int(os.environ.get("PADDLE_OCR_DPI", "150"))

_ocr_vl:    "object | None" = None
_load_lock = threading.Lock()

# Dedicated single-thread executor — one GPU OCR job at a time.
_OCR_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paddle_ocr_worker")


def _get_ocr_model():
    global _ocr_vl
    if _ocr_vl is None:
        with _load_lock:
            if _ocr_vl is None:  # re-check inside the lock
                from paddleocr import PaddleOCRVL
                logger.info("[paddle_ocr_utils] Loading PaddleOCR-VL 1.5 ...")
                t0 = time.perf_counter()
                _ocr_vl = PaddleOCRVL("v1.5")
                logger.info(
                    f"[paddle_ocr_utils] PaddleOCR-VL ready "
                    f"({time.perf_counter() - t0:.2f}s)"
                )
    return _ocr_vl


def _page_to_image(page: fitz.Page, dpi: int = OCR_DPI) -> np.ndarray:
    """Render a fitz page to a C-contiguous uint8 RGB numpy array."""
    pix = page.get_pixmap(dpi=dpi)
    img = np.ascontiguousarray(
        np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    )
    if pix.n == 4:
        img = img[:, :, :3]
    return img


_OCR_TEXT_KEYS = ("rec_text", "text")

_OCR_JUNK_MARKERS = (
    "numpy.ndarray",
    "layout_det",
    "rec_score",
    "table_res",
    "input_path",
    "model_settings",
    "parsing_res",
    "spotting_res",
    "page_id",
)


def _extract_ocr_text(res) -> str:
    """
    Safely extract the human-readable OCR text from a single PaddleOCR-VL
    result item, ignoring internal metadata/debug fields.
    """
    if res is None:
        return ""

    if isinstance(res, str):
        s = res.strip()
        if any(marker in s for marker in _OCR_JUNK_MARKERS):
            return ""
        return s

    if isinstance(res, dict):
        for key in _OCR_TEXT_KEYS:
            if key in res:
                val = res[key]
                return val.strip() if isinstance(val, str) else ""
        if "res" in res:
            return _extract_ocr_text(res["res"])
        return ""

    for key in _OCR_TEXT_KEYS:
        if hasattr(res, key):
            val = getattr(res, key)
            return val.strip() if isinstance(val, str) else ""

    if hasattr(res, "res"):
        return _extract_ocr_text(res.res)

    return ""


def _ocr_predict(img: np.ndarray) -> list:
    """Runs in _OCR_EXECUTOR so it can be cancelled via future timeout."""
    return _get_ocr_model().predict(img)


def _empty_cuda_cache():
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
    except Exception:
        pass


def run_paddle_ocr_on_pdf(file_path: str, max_pages: int | None = None) -> dict:
    """
    Run every page of a PDF through local PaddleOCR-VL 1.5 unconditionally
    (no native-text-first check — this endpoint exists specifically to test
    the PaddleOCR-VL engine itself).

    Returns:
      {
        "total_pages": int,
        "pages_processed": int,
        "pages": [
          {"page": 1, "text": "...", "elapsed_s": 1.23, "timed_out": False},
          ...
        ],
        "total_time_s": float,
      }
    """
    t_start = time.perf_counter()

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise ValueError(f"Could not open PDF: {e}")

    total_pages      = len(doc)
    pages_to_process = total_pages if max_pages is None else min(total_pages, max_pages)

    logger.info(
        f"[paddle_ocr_utils] Running PaddleOCR-VL on {pages_to_process}/{total_pages} page(s) "
        f"(timeout={OCR_PAGE_TIMEOUT}s/page, dpi={OCR_DPI})"
    )

    results = []
    for idx in range(pages_to_process):
        page_num = idx + 1
        img      = _page_to_image(doc[idx])

        t_page    = time.perf_counter()
        text      = ""
        timed_out = False

        try:
            future  = _OCR_EXECUTOR.submit(_ocr_predict, img)
            preds   = future.result(timeout=OCR_PAGE_TIMEOUT)
            parts   = [t for t in (_extract_ocr_text(r) for r in preds) if t]
            text    = clean_text("\n".join(parts))
            _empty_cuda_cache()

        except FuturesTimeoutError:
            timed_out = True
            future.cancel()
            _empty_cuda_cache()
            logger.error(
                f"[paddle_ocr_utils] Page {page_num}: TIMEOUT "
                f"(> {OCR_PAGE_TIMEOUT}s)"
            )

        except Exception as e:
            _empty_cuda_cache()
            logger.warning(f"[paddle_ocr_utils] Page {page_num}: OCR failed — {e}")

        elapsed = time.perf_counter() - t_page
        logger.info(
            f"[paddle_ocr_utils] Page {page_num}: {len(text)} char(s) "
            f"({elapsed:.2f}s){' TIMEOUT' if timed_out else ''}"
        )

        results.append({
            "page":      page_num,
            "text":      text,
            "elapsed_s": round(elapsed, 3),
            "timed_out": timed_out,
        })

    doc.close()
    total_time = time.perf_counter() - t_start
    logger.info(
        f"[paddle_ocr_utils] Done — {pages_to_process} page(s) in {total_time:.2f}s"
    )

    return {
        "total_pages":     total_pages,
        "pages_processed": pages_to_process,
        "pages":           results,
        "total_time_s":    round(total_time, 3),
    }
