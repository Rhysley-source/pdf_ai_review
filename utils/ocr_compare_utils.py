import fitz
import numpy as np
import base64
from io import BytesIO
from PIL import Image


def pdf_to_images(pdf_bytes: bytes, dpi: int = 150):
    """
    Convert PDF bytes → list of numpy RGB images
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.h, pix.w, pix.n
        )

        if pix.n == 4:
            img = img[:, :, :3]

        images.append(img)

    return images


def img_to_base64(img: np.ndarray) -> str:
    """
    Convert numpy image → base64 PNG
    """
    pil = Image.fromarray(img)
    buffer = BytesIO()
    pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def run_paddleocr(img, paddle_ocr):
    """
    PaddleOCR-VL inference wrapper
    """
    res = paddle_ocr.predict(img)

    return "\n".join(
        r.get("rec_text", "")
        for r in res
        if isinstance(r, dict)
    ).strip()


def run_openai_vision(base64_img: str, openai_client):
    """
    OpenAI Vision OCR (gpt-4o-mini)
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract all text from this image accurately."
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