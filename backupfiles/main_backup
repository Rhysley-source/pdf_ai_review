# from fastapi import FastAPI, UploadFile, File, Query
# import os, json, re
# from pdf_utils import extract_text_from_pdf, chunk_text
# from ai_model import generate_analysis

# app = FastAPI()

# UPLOAD_FOLDER = "temp"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# def clean_text(text: str) -> str:
#     """Clean PDF text for better model analysis"""
#     text = text.replace("\n\n", "\n")  # remove extra blank lines
#     text = text.strip()
#     return text


# def extract_json(text):
#     try:
#         # Try JSON first
#         match = re.search(r"\{.*\}", text, re.DOTALL)
#         if match:
#             return json.loads(match.group())
#     except:
#         pass

#     # Fallback: parse text format
#     summary = ""
#     positive = []
#     negative = []
#     avoid = []

#     sections = text.split("\n")

#     current = None
#     for line in sections:
#         line = line.strip()

#         if "Summary" in line:
#             current = "summary"
#             continue
#         elif "Positive" in line:
#             current = "positive"
#             continue
#         elif "Negative" in line:
#             current = "negative"
#             continue
#         elif "Avoid" in line:
#             current = "avoid"
#             continue

#         if current == "summary":
#             summary += line + " "

#         elif current == "positive" and line:
#             positive.append(line)

#         elif current == "negative" and line:
#             negative.append(line)

#         elif current == "avoid" and line:
#             avoid.append(line)

#     return {
#         "summary": summary.strip(),
#         "positive_points": positive,
#         "negative_points": negative,
#         "how_to_avoid": avoid
#     }


# @app.post("/analyze")
# async def analyze_pdf(
#     file: UploadFile = File(...),
#     analysis_type: int = Query(
#         0, ge=0, le=4,
#         description="0=all, 1=summary, 2=positive, 3=negative, 4=how_to_avoid"
#     )
# ):
#     """
#     Advanced PDF analysis with chunking and selective section output.
#     """

#     # Save uploaded PDF
#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     with open(file_path, "wb") as f:
#         f.write(await file.read())

#     # Extract text from PDF
#     text = extract_text_from_pdf(file_path)
#     text = clean_text(text)

#     # Chunk the text
#     chunks = chunk_text(text)

#     results = []

#     for chunk in chunks:
#         ai_response = generate_analysis(chunk)
#         print(ai_response)   # DEBUG
#         parsed = extract_json(ai_response)
#         results.append(parsed)

#     # Merge results
#     final_output = {
#         "summary": " ".join([r.get("summary", "") for r in results]),
#         "positive_points": sum([r.get("positive_points", []) for r in results], []),
#         "negative_points": sum([r.get("negative_points", []) for r in results], []),
#         "how_to_avoid": sum([r.get("how_to_avoid", []) for r in results], [])
#     }

#     # Return based on analysis_type
#     if analysis_type == 1:
#         return {"summary": final_output["summary"]}
#     elif analysis_type == 2:
#         return {"positive_points": final_output["positive_points"]}
#     elif analysis_type == 3:
#         return {"negative_points": final_output["negative_points"]}
#     elif analysis_type == 4:
#         return {"how_to_avoid": final_output["how_to_avoid"]}

#     return final_output






# def extract_json(text: str):
#     """Extract strict JSON from model output"""
#     try:
#         match = re.search(r"\{.*\}", text, re.DOTALL)
#         if match:
#             return json.loads(match.group())
#     except Exception as e:
#         print("JSON parse error:", e)

#     # Safe fallback
#     return {
#         "overview": "",
#         "summary": "",
#         "highlights": []
#     }


# def extract_json(text: str):
#     """Extract and repair JSON from model output"""
#     try:
#         # 1. Find the first '{' and last '}' to strip any preamble or rambling
#         start = text.find('{')
#         end = text.rfind('}')
#         if start != -1 and end != -1:
#             json_str = text[start:end+1]
            
#             # 2. Fix common "escape" errors (replaces \ with \\ unless it's a valid escape)
#             # This fixes the "Invalid \escape" error you saw
#             json_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', json_str)
            
#             return json.loads(json_str)
#     except Exception as e:
#         print(f"JSON repair failed: {e}")

#     # Fallback to manual regex if JSON is totally mangled
#     return {
#         "overview": re.search(r'"overview":\s*"(.*?)"', text).group(1) if '"overview"' in text else "",
#         "summary": re.search(r'"summary":\s*"(.*?)"', text).group(1) if '"summary"' in text else "",
#         "highlights": re.findall(r'"highlights":\s*\[(.*?)\]', text, re.DOTALL) or []
#     }



#==================this is final code ===========================

# from fastapi import FastAPI, UploadFile, File, Query
# import os
# import json
# import re
# from pdf_utils import extract_text_from_pdf, chunk_text
# from ai_model import generate_analysis

# app = FastAPI()

# UPLOAD_FOLDER = "temp"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# def clean_text(text: str) -> str:
#     """Clean extracted PDF text"""
#     text = text.replace("\n\n", "\n")
#     return text.strip()

# def extract_json(text: str):
#     """Cleanly extract and structure JSON from Mistral output"""
#     try:
#         # 1. Clean up "Markdown-isms" like ```json ... ```
#         text = text.replace("```json", "").replace("```", "").strip()
        
#         # 2. Extract content between first { and last }
#         match = re.search(r"\{.*\}", text, re.DOTALL)
#         if match:
#             clean_str = match.group()
#             # Fix illegal backslashes often found in model outputs
#             clean_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', clean_str)
#             data = json.loads(clean_str)

#             # 3. Post-Process 'highlights' to remove double-quotes if they exist
#             if "highlights" in data and isinstance(data["highlights"], list):
#                 # Flatten the list and strip inner escaped quotes
#                 cleaned_highlights = []
#                 for item in data["highlights"]:
#                     # Split if the model put everything in one string separated by commas
#                     parts = item.split('", "') if '", "' in item else [item]
#                     for p in parts:
#                         cleaned_highlights.append(p.replace('"', '').strip())
#                 data["highlights"] = cleaned_highlights
            
#             return data
#     except Exception as e:
#         print(f"Extraction Error: {e}")

#     return {"overview": "", "summary": "", "highlights": []}




# @app.post("/analyze")
# async def analyze_pdf(
#     file: UploadFile = File(...),
#     analysis_type: int = Query(
#         0,
#         ge=0,
#         le=3,
#         description="0=all, 1=overview, 2=summary, 3=highlights"
#     )
# ):
#     """
#     Analyze uploaded PDF and return:
#     - overview
#     - summary
#     - highlights
#     """

#     # Save uploaded file
#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)

#     with open(file_path, "wb") as f:
#         f.write(await file.read())

#     # Extract text from PDF
#     text = extract_text_from_pdf(file_path)
#     text = clean_text(text)

#     # Split into chunks (for large PDFs)
#     chunks = chunk_text(text)

#     results = []

#     for chunk in chunks:
#         ai_response = generate_analysis(chunk)
#         parsed = extract_json(ai_response)
#         results.append(parsed)

#     # Merge results from all chunks
#     combined_overview = " ".join(
#         [r.get("overview", "") for r in results]
#     ).strip()

#     combined_summary = " ".join(
#         [r.get("summary", "") for r in results]
#     ).strip()

#     combined_highlights = list(set(
#         sum([r.get("highlights", []) for r in results], [])
#     ))

#     final_output = {
#         "overview": combined_overview,
#         "summary": combined_summary,
#         "highlights": combined_highlights
#     }

#     # Return selected section if requested
#     if analysis_type == 1:
#         return {"overview": final_output["overview"]}
#     elif analysis_type == 2:
#         return {"summary": final_output["summary"]}
#     elif analysis_type == 3:
#         return {"highlights": final_output["highlights"]}

#     return final_output




# modified code 


# from fastapi import FastAPI, UploadFile, File, Query, HTTPException
# import os
# import uuid
# import json
# import re
# import logging
# from pdf_utils import extract_text_from_pdf, chunk_text
# from ai_model import generate_analysis

# logger = logging.getLogger(__name__)

# app = FastAPI()

# UPLOAD_FOLDER = "temp"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# def clean_text(text: str) -> str:
#     """Clean extracted PDF text"""
#     text = text.replace("\n\n", "\n")
#     return text.strip()

# def extract_json(text: str):
#     """Cleanly extract and structure JSON from Mistral output"""
#     try:
#         # 1. Clean up "Markdown-isms" like ```json ... ```
#         text = text.replace("```json", "").replace("```", "").strip()
        
#         # 2. Extract content between first { and last }
#         match = re.search(r"\{.*\}", text, re.DOTALL)
#         if match:
#             clean_str = match.group()
#             # Fix illegal backslashes often found in model outputs
#             clean_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', clean_str)
#             data = json.loads(clean_str)

#             # 3. Post-Process 'highlights' to remove double-quotes if they exist
#             if "highlights" in data and isinstance(data["highlights"], list):
#                 # Flatten the list and strip inner escaped quotes
#                 cleaned_highlights = []
#                 for item in data["highlights"]:
#                     # Split if the model put everything in one string separated by commas
#                     parts = item.split('", "') if '", "' in item else [item]
#                     for p in parts:
#                         cleaned_highlights.append(p.replace('"', '').strip())
#                 data["highlights"] = cleaned_highlights
            
#             return data
#     except Exception as e:
#         logger.error(f"Extraction Error: {e}")

#     return {"overview": "", "summary": "", "highlights": []}


# @app.post("/analyze")
# async def analyze_pdf(
#     file: UploadFile = File(...),
#     analysis_type: int = Query(
#         0,
#         ge=0,
#         le=3,
#         description="0=all, 1=overview, 2=summary, 3=highlights"
#     )
# ):
#     """
#     Analyze uploaded PDF and return:
#     - overview
#     - summary
#     - highlights
#     """

#     # Validate file type
#     if not file.filename or not file.filename.lower().endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

#     # Use a random UUID filename to prevent path traversal attacks
#     safe_name = f"{uuid.uuid4()}.pdf"
#     file_path = os.path.join(UPLOAD_FOLDER, safe_name)

#     try:
#         # Save uploaded file
#         with open(file_path, "wb") as f:
#             f.write(await file.read())

#         # Extract text from PDF
#         text = extract_text_from_pdf(file_path)
#         text = clean_text(text)

#         # Reject image-only or empty PDFs early
#         if not text:
#             raise HTTPException(
#                 status_code=422,
#                 detail="No extractable text found in PDF. The file may be image-only or empty."
#             )

#         # Split into chunks (for large PDFs)
#         chunks = chunk_text(text)

#         results = []

#         for chunk in chunks:
#             ai_response = await generate_analysis(chunk)
#             parsed = extract_json(ai_response)
#             results.append(parsed)

#     finally:
#         # Always clean up the uploaded file from disk
#         if os.path.exists(file_path):
#             os.remove(file_path)

#     # Merge results from all chunks
#     combined_overview = " ".join(
#         [r.get("overview", "") for r in results]
#     ).strip()

#     combined_summary = " ".join(
#         [r.get("summary", "") for r in results]
#     ).strip()

#     combined_highlights = list(set(
#         sum([r.get("highlights", []) for r in results], [])
#     ))

#     final_output = {
#         "overview": combined_overview,
#         "summary": combined_summary,
#         "highlights": combined_highlights
#     }

#     # Return selected section if requested
#     if analysis_type == 1:
#         return {"overview": final_output["overview"]}
#     elif analysis_type == 2:
#         return {"summary": final_output["summary"]}
#     elif analysis_type == 3:
#         return {"highlights": final_output["highlights"]}

#     return final_output




from fastapi import FastAPI, UploadFile, File, Query, HTTPException
import os
import uuid
import json
import re
import logging
from pdf_utils import extract_text_from_pdf, chunk_text
from ai_model import generate_analysis

logger = logging.getLogger(__name__)

app = FastAPI()

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def clean_text(text: str) -> str:
    """Clean extracted PDF text"""
    text = text.replace("\n\n", "\n")
    return text.strip()

def _fix_json_string(raw: str) -> str:
    """
    Apply a sequence of targeted repairs to common Mistral JSON output problems,
    without altering the structure or values of well-formed output.
    """
    # Remove markdown code fences
    raw = raw.replace("```json", "").replace("```", "").strip()

    # Normalise Windows line endings
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Fix illegal (non-JSON) backslash escapes, e.g. \' \, \: \. etc.
    # Keep valid escapes: \" \\ \/ \b \f \n \r \t \uXXXX
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)

    # Remove ASCII control characters (0x00-0x1F) except \n \r \t which are valid in JSON strings
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)

    return raw


def _extract_field_by_regex(text: str, field: str) -> str:
    """
    Fallback: extract a single string field value directly with regex
    when the whole JSON block can't be parsed.
    """
    pattern = rf'"{field}"\s*:\s*"(.*?)"(?=\s*[,}}])'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _extract_highlights_by_regex(text: str) -> list:
    """
    Fallback: extract highlights array items directly with regex.
    """
    match = re.search(r'"highlights"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        return []
    inner = match.group(1)
    items = re.findall(r'"(.*?)"(?=\s*[,\]])', inner, re.DOTALL)
    return [i.strip() for i in items if i.strip()]


def _postprocess_highlights(data: dict) -> dict:
    """
    Post-process highlights list: flatten items the model packed into a single
    comma-separated string, and strip stray quote characters.
    """
    if "highlights" in data and isinstance(data["highlights"], list):
        cleaned = []
        for item in data["highlights"]:
            parts = item.split('", "') if '", "' in item else [item]
            for p in parts:
                cleaned.append(p.replace('"', '').strip())
        data["highlights"] = [h for h in cleaned if h]
    return data


def extract_json(text: str) -> dict:
    """
    Robustly extract and structure JSON from Mistral output using three
    fallback strategies so a single malformed character never silently
    returns an empty result.

    Strategy 1 — direct parse after cleaning
    Strategy 2 — isolate outermost { ... } block, then parse; on failure
                 attempt a conservative in-string quote-escape repair
    Strategy 3 — per-field regex extraction as last resort
    """
    empty = {"overview": "", "summary": "", "highlights": []}

    if not text or not text.strip():
        return empty

    cleaned = _fix_json_string(text)

    # --- Strategy 1: parse the whole cleaned string directly ---
    try:
        data = json.loads(cleaned)
        return _postprocess_highlights(data)
    except json.JSONDecodeError:
        pass

    # --- Strategy 2: isolate the outermost { ... } block ---
    brace_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if brace_match:
        candidate = brace_match.group()
        try:
            data = json.loads(candidate)
            return _postprocess_highlights(data)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON block parse failed ({e}), trying character-level repair")

            # Sub-strategy: attempt to repair unescaped double-quotes inside string values.
            # Replace any " that is NOT preceded by \ and NOT a structural delimiter
            # (i.e. not after : [ , { or before ] } ,) with \".
            # This is intentionally conservative — only fixes the most common model error.
            repaired = re.sub(
                r'(?<=[^\\])"(?=[^,\]}\n:}{\[])',
                r'\\"',
                candidate
            )
            try:
                data = json.loads(repaired)
                return _postprocess_highlights(data)
            except json.JSONDecodeError:
                pass

    # --- Strategy 3: regex field extraction (last resort) ---
    logger.error("Extraction Error: all JSON parse strategies failed, falling back to regex extraction")
    overview = _extract_field_by_regex(cleaned, "overview")
    summary = _extract_field_by_regex(cleaned, "summary")
    highlights = _extract_highlights_by_regex(cleaned)

    if overview or summary or highlights:
        return {"overview": overview, "summary": summary, "highlights": highlights}

    return empty


@app.post("/analyze")
async def analyze_pdf(
    file: UploadFile = File(...),
    analysis_type: int = Query(
        0,
        ge=0,
        le=3,
        description="0=all, 1=overview, 2=summary, 3=highlights"
    )
):
    """
    Analyze uploaded PDF and return:
    - overview
    - summary
    - highlights
    """

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Use a random UUID filename to prevent path traversal attacks
    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Extract text from PDF
        text = extract_text_from_pdf(file_path)
        text = clean_text(text)

        # Reject image-only or empty PDFs early
        if not text:
            raise HTTPException(
                status_code=422,
                detail="No extractable text found in PDF. The file may be image-only or empty."
            )

        # Split into chunks (for large PDFs)
        chunks = chunk_text(text)

        results = []

        for chunk in chunks:
            ai_response = await generate_analysis(chunk)
            parsed = extract_json(ai_response)
            results.append(parsed)

    finally:
        # Always clean up the uploaded file from disk
        if os.path.exists(file_path):
            os.remove(file_path)

    # Merge results from all chunks
    combined_overview = " ".join(
        [r.get("overview", "") for r in results]
    ).strip()

    combined_summary = " ".join(
        [r.get("summary", "") for r in results]
    ).strip()

    combined_highlights = list(set(
        sum([r.get("highlights", []) for r in results], [])
    ))

    final_output = {
        "overview": combined_overview,
        "summary": combined_summary,
        "highlights": combined_highlights
    }

    # Return selected section if requested
    if analysis_type == 1:
        return {"overview": final_output["overview"]}
    elif analysis_type == 2:
        return {"summary": final_output["summary"]}
    elif analysis_type == 3:
        return {"highlights": final_output["highlights"]}

    return final_output