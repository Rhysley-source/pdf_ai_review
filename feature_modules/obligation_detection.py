"""
obligation_detection.py

Detects and extracts obligations from legal and business documents.
Identifies both positive obligations (MUST DO) and negative obligations (MUST NOT DO).
"""

import logging
import time
import uuid

from utils.llm_utils import run_llm_with_tokens  
from utils.json_utils import extract_json_raw as extract_json_from_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------

async def analyze_document_obligations(text: str) -> tuple[dict, int, int]:

    request_id = str(uuid.uuid4())[:8]
    t_start = time.perf_counter()

    total_in_tokens = 0
    total_out_tokens = 0

    logger.info(f"[{request_id}] ── OBLIGATION DETECTION ─────────────────────")

    try:
        # ---------------------------
        # STEP 1 — CLASSIFICATION
        # ---------------------------
        logger.info(f"[{request_id}] Step 1/4 — classifying document")

        t_classify = time.perf_counter()

        classify_prompt = f"""
Classify this document into ONE:

nda
rent_agreement
employment_contract
legal_notice
general_contract
other

Return ONLY one word.

Text:
\"\"\"{text[:1500]}\"\"\"
"""

        doc_type_raw, in_tok1, out_tok1 = await run_llm_with_tokens(text[:1500], classify_prompt)

        total_in_tokens += in_tok1
        total_out_tokens += out_tok1

        doc_type = doc_type_raw.lower().strip()

        if "nda" in doc_type:
            doc_type = "nda"
        elif "rent" in doc_type:
            doc_type = "rent_agreement"
        elif "employment" in doc_type:
            doc_type = "employment_contract"
        elif "notice" in doc_type or "summons" in doc_type:
            doc_type = "legal_notice"
        elif "contract" in doc_type:
            doc_type = "general_contract"
        else:
            doc_type = "other"

        logger.info(
            f"[{request_id}] Step 1/4 — classified as '{doc_type}' "
            f"({time.perf_counter()-t_classify:.2f}s)"
        )

        # ---------------------------
        # STEP 2 — BUILD PROMPT
        # ---------------------------
        logger.info(f"[{request_id}] Step 2/4 — preparing extraction")

        base_instruction = """
Extract ALL obligations from the document.

IMPORTANT:
- Be exhaustive
- Do NOT miss implicit obligations
- Split multiple obligations in one sentence
- Clearly identify who owes obligation to whom
"""

        type_focus = {
            "nda": "Focus on confidentiality, non-disclosure, permitted use, data return, duration.",
            "rent_agreement": "Focus on rent payment, maintenance, restrictions, penalties, landlord duties.",
            "employment_contract": "Focus on duties, confidentiality, notice period, non-compete.",
            "legal_notice": "Focus on deadlines, mandatory actions, penalties, legal consequences.",
            "general_contract": "Focus on all contractual obligations across parties.",
            "other": "Extract any obligation-like statement."
        }

        prompt = f"""
You are an expert legal obligation extraction system.

{base_instruction}

{type_focus.get(doc_type)}

-----------------------------------

For EACH obligation extract:

- obligation_id (POS-001, NEG-001...)
- type → must_do or must_not_do
- party_responsible
- counterparty
- relationship → "owes_to"

- action (single verb)
- normalized_action (normalized verb)
- object
- condition

- obligation_text (clear sentence)

- deadline
- normalized_deadline (YYYY-MM-DD or null)

- schedule:
  {{
    "is_recurring": boolean,
    "frequency": daily/weekly/monthly/yearly/null,
    "interval": number/null,
    "day_of_month": number/null,
    "day_of_week": string/null,
    "time": null,
    "start_date": null,
    "end_date": null
  }}

- duration
- consequence
- clause_reference

- priority (High/Medium/Low)
- confidence (0 to 1)

- source_text (exact sentence)

-----------------------------------

OUTPUT FORMAT (STRICT JSON ONLY):

{{
  "summary": {{
    "total_obligations": 0,
    "positive": 0,
    "negative": 0,
    "high_priority": 0
  }},
  "parties_identified": [],
  "obligations": [],
  "high_priority_obligation_ids": [],
  "obligations_by_party": {{}}
}}

Document:
---
{text[:80_000]}
---
"""

        # ---------------------------
        # STEP 3 — INFERENCE
        # ---------------------------
        logger.info(f"[{request_id}] Step 3/4 — running inference")

        t_infer = time.perf_counter()

        raw_output, in_tok2, out_tok2 = await run_llm_with_tokens(text, prompt)

        total_in_tokens += in_tok2
        total_out_tokens += out_tok2

        logger.info(
            f"[{request_id}] Step 3/4 — done ({time.perf_counter()-t_infer:.2f}s)"
        )

        analysis = extract_json_from_text(raw_output)

        obligations = analysis.get("obligations", [])

        logger.info(
            f"[{request_id}] Step 4/4 — extracted {len(obligations)} obligations"
        )

        # ---------------------------
        # RETRY IF WEAK
        # ---------------------------
        if len(obligations) < 3:
            logger.warning(f"[{request_id}] weak extraction — retrying")

            retry_prompt = prompt.replace(
                "Be exhaustive",
                "Be EXTREMELY exhaustive. Do not miss ANY obligation."
            )

            t_retry = time.perf_counter()

            raw_retry, in_tok3, out_tok3 = await run_llm_with_tokens(text, retry_prompt)

            total_in_tokens += in_tok3
            total_out_tokens += out_tok3

            retry_analysis = extract_json_from_text(raw_retry)
            retry_obligations = retry_analysis.get("obligations", [])

            logger.info(
                f"[{request_id}] retry — {len(retry_obligations)} obligations "
                f"({time.perf_counter()-t_retry:.2f}s)"
            )

            if len(retry_obligations) > len(obligations):
                logger.info(f"[{request_id}] retry improved results")
                analysis = retry_analysis
                obligations = retry_obligations

        # ---------------------------
        # POST PROCESSING
        # ---------------------------
        positive = sum(1 for o in obligations if o.get("type") == "must_do")
        negative = sum(1 for o in obligations if o.get("type") == "must_not_do")
        high_priority = [o for o in obligations if o.get("priority") == "High"]

        analysis["summary"] = {
            "total_obligations": len(obligations),
            "positive": positive,
            "negative": negative,
            "high_priority": len(high_priority)
        }

        analysis["high_priority_obligation_ids"] = [
            o.get("obligation_id") for o in high_priority if o.get("obligation_id")
        ]

        party_map = {}
        for o in obligations:
            party = o.get("party_responsible", "Unknown")
            party_map.setdefault(party, []).append(o.get("obligation_id"))

        analysis["obligations_by_party"] = party_map

        # ---------------------------
        # FINAL LOGS (MATCH /analyze)
        # ---------------------------
        logger.info(
            f"[{request_id}] obligations={len(obligations)} "
            f"tokens={total_in_tokens}in/{total_out_tokens}out"
        )

        elapsed = time.perf_counter() - t_start

        logger.info(f"[{request_id}] ── COMPLETE — {elapsed:.2f}s ──────")

        return {
            "status": "success",
            "analysis_type": "obligation_detection",
            "document_type": doc_type,
            "data": analysis
        }, total_in_tokens, total_out_tokens

    except Exception as e:
        logger.exception(f"[{request_id}] unhandled error: {e}")
        raise