import base64
import numpy as np
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF

# ----------------------------
# PaddleOCRVL (singleton)
# ----------------------------
from paddleocr import PaddleOCRVL
paddle_ocr = PaddleOCRVL()

# ----------------------------
# OpenAI client (must exist)
# ----------------------------
from llm_model.openai_client import openai_client


# =========================================================
# PDF → Images
# =========================================================
def pdf_to_images(pdf_bytes: bytes, dpi: int = 150):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    for page in doc:
        pix = page.get_pixmap(dpi=dpi)

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.h, pix.w, pix.n
        )

        # remove alpha channel if exists
        if pix.n == 4:
            img = img[:, :, :3]

        images.append(img)

    return images


# =========================================================
# Image → Base64
# =========================================================
def img_to_base64(img: np.ndarray) -> str:
    pil = Image.fromarray(img)
    buffer = BytesIO()
    pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# =========================================================
# PaddleOCR VL
# =========================================================
def run_paddleocr(img: np.ndarray) -> str:
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
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
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
            }],
            temperature=0
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"[OpenAI Vision Error] {str(e)}"


# =========================================================
# HYBRID OCR (BEST)
# =========================================================
def hybrid_ocr(img: np.ndarray) -> str:
    text = run_paddleocr(img)

    if text and len(text.strip()) > 5:
        return text

    base64_img = img_to_base64(img)
    return run_openai_vision(base64_img)