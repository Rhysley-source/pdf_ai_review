import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI
from paddleocr import PaddleOCRVL

# =========================================================
# ENV LOAD (must be at top)
# =========================================================
load_dotenv()

# =========================================================
# SINGLETON CLIENTS
# =========================================================

# OpenAI client
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# PaddleOCR-VL (heavy model → load once)
paddle_ocr = PaddleOCRVL()


# =========================================================
# PDF → Images
# =========================================================
def pdf_to_images(pdf_bytes: bytes, dpi: int = 150):
    """
    Convert PDF bytes to list of numpy images
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    for page in doc:
        pix = page.get_pixmap(dpi=dpi)

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.h, pix.w, pix.n
        )

        # Remove alpha channel if present
        if pix.n == 4:
            img = img[:, :, :3]

        images.append(img)

    return images


# =========================================================
# Image → Base64
# =========================================================
def img_to_base64(img: np.ndarray) -> str:
    """
    Convert numpy image to base64 string
    """
    pil = Image.fromarray(img)
    buffer = BytesIO()
    pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# =========================================================
# PaddleOCR VL
# =========================================================
def run_paddleocr(img: np.ndarray) -> str:
    """
    Run PaddleOCR-VL on image
    """
    try:
        res = paddle_ocr.predict(img)

        return "\n".join(
            r.get("rec_text", "")
            for r in res
            if isinstance(r, dict)
        ).strip()

    except Exception as e:
        return f"[PaddleOCR Error] {str(e)}"


# =========================================================
# OpenAI Vision OCR
# =========================================================
def run_openai_vision(base64_img: str) -> str:
    """
    Run OpenAI Vision OCR (fallback / comparison)
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this image accurately. Preserve layout."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_img}"
                            }
                        }
                    ]
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"[OpenAI Vision Error] {str(e)}"


# =========================================================
# HYBRID OCR (BEST STRATEGY)
# =========================================================
def hybrid_ocr(img: np.ndarray) -> str:
    """
    First try PaddleOCR (fast, local)
    Fallback to OpenAI Vision if weak/empty
    """
    text = run_paddleocr(img)

    # If Paddle result is weak → fallback
    if not text or len(text.strip()) < 5:
        base64_img = img_to_base64(img)
        return run_openai_vision(base64_img)

    return text