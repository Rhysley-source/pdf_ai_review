import io
import os
import subprocess
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from auth import verify_api_key
from docx import Document
from docx.shared import mm

router = APIRouter()


@router.post("/convert/docx-to-pdf")
async def convert_docx_to_pdf(
    file: UploadFile = File(...),
    _: None = Depends(verify_api_key),
):
    """
    Accepts a .docx file and returns a converted PDF using LibreOffice headless.
    Forces A4 paper size.
    """
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported")

    with tempfile.TemporaryDirectory() as tmpdir:
        docx_path = os.path.join(tmpdir, file.filename)
        # Read the file content once
        content = await file.read()
        
        # Write initial docx
        with open(docx_path, "wb") as f:
            f.write(content)

        # Force A4 size using python-docx
        try:
            doc = Document(docx_path)
            for section in doc.sections:
                section.page_height = mm(297)
                section.page_width = mm(210)
            doc.save(docx_path)
        except Exception as e:
            # If docx editing fails, we still try to convert the original
            pass

        result = subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to", "pdf",
                "--outdir", tmpdir,
                docx_path,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Conversion failed: {result.stderr.strip()}",
            )

        pdf_filename = os.path.splitext(file.filename)[0] + ".pdf"
        pdf_path = os.path.join(tmpdir, pdf_filename)

        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=500, detail="PDF file was not generated")

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{pdf_filename}"'},
    )
