import io
import os
import subprocess
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from auth import verify_api_key

router = APIRouter()


@router.post("/convert/docx-to-pdf")
async def convert_docx_to_pdf(
    file: UploadFile = File(...),
    _: None = Depends(verify_api_key),
):
    """
    Accepts a .docx file and returns a converted PDF using LibreOffice headless.
    """
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported")

    with tempfile.TemporaryDirectory() as tmpdir:
        docx_path = os.path.join(tmpdir, file.filename)
        with open(docx_path, "wb") as f:
            f.write(await file.read())

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


@router.post("/convert/doc-to-docx")
async def convert_doc_to_docx(
    file: UploadFile = File(...),
    _: None = Depends(verify_api_key),
):
    """
    Accepts a legacy .doc file and returns a converted .docx using LibreOffice headless.
    """
    if not file.filename or not file.filename.lower().endswith(".doc"):
        raise HTTPException(status_code=400, detail="Only .doc files are supported")

    with tempfile.TemporaryDirectory() as tmpdir:
        doc_path = os.path.join(tmpdir, file.filename)
        with open(doc_path, "wb") as f:
            f.write(await file.read())

        result = subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to", "docx",
                "--outdir", tmpdir,
                doc_path,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Conversion failed: {result.stderr.strip()}",
            )

        docx_filename = os.path.splitext(file.filename)[0] + ".docx"
        docx_path = os.path.join(tmpdir, docx_filename)

        if not os.path.exists(docx_path):
            raise HTTPException(status_code=500, detail="DOCX file was not generated")

        with open(docx_path, "rb") as f:
            docx_bytes = f.read()

    return StreamingResponse(
        io.BytesIO(docx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{docx_filename}"'},
    )
