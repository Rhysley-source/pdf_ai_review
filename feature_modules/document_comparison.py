"""
document_comparison.py — LLM enrichment prompt upgrade
 
Improvements to the LLM prompt (applying patterns from prompts.py + obligation_detection.py):
  - Role persona: "senior legal analyst specialising in contract risk review"
  - Per-clause summary instruction is explicit and structured
  - Semantic insights demand specific values, percentages, durations
  - Final insight must name the favoured party
  - Recommendation is actionable with named clauses
  - Retry uses a worked example, not a useless empty prompt
  - Document text passed as focused snippets not 3000-char truncations
"""

import logging
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from difflib import SequenceMatcher
 
from llm_model.ai_model import run_llm_raw_json, run_llm_with_tokens
from utils.json_utils import extract_json_raw as extract_json_from_text
 
logger = logging.getLogger(__name__)
 
HIGH_RISK_CLAUSES = {
    "payment_terms","liability","termination","indemnification",
    "penalties","pricing","intellectual_property",
}
MEDIUM_RISK_CLAUSES = {
    "contract_term","renewal_conditions","scope_of_work",
    "delivery_timeline","non_compete","warranties","force_majeure","data_protection",
}
MIN_CLAUSE_LENGTH = 80   # filters sentence fragments from extracted clauses
MATCH_THRESHOLD   = 0.55

# Obligation doc-type taxonomy (mirrors obligation_detection.py)
_OBL_DOC_TYPES = {"nda", "rent_agreement", "employment_contract",
                   "legal_notice", "general_contract", "other"}

# Map clause_extraction doc types → obligation doc types
_CLAUSE_TO_OBL_TYPE = {
    "contract":   "general_contract",
    "nda":        "nda",
    "employment": "employment_contract",
    "lease":      "rent_agreement",
    "invoice":    "other",
    "resume":     "other",
    "other":      "other",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import re as _re

def _norm(t):
    """Lowercase, strip, and remove bullet-numbering so format-only changes don't count as diffs."""
    t = (t or "").lower().strip()
    # Remove numbered bullet prefixes like "1.", "2.", "1)", "2)" with optional unicode spaces
    t = _re.sub(r'(?<!\w)\d+[\.\)]\s*[\u200b\u200c\u200d\ufeff]?\s*', ' ', t)
    # Remove zero-width characters
    t = _re.sub(r'[\u200b\u200c\u200d\ufeff]', '', t)
    # Collapse multiple spaces
    t = _re.sub(r'\s+', ' ', t).strip()
    return t

def _sim(a,b): return SequenceMatcher(None,_norm(a),_norm(b)).ratio()
def _ksim(k1,k2): return SequenceMatcher(None,k1.replace("_"," "),k2.replace("_"," ")).ratio()
def _csim(k1,v1,k2,v2): return 0.5*_ksim(k1,k2)+0.5*_sim(v1[:400],v2[:400])
def _sev(k):
    if k in HIGH_RISK_CLAUSES:   return "high"
    if k in MEDIUM_RISK_CLAUSES: return "medium"
    return "low"
def _clean(c): return {k:v.strip() for k,v in (c or {}).items() if isinstance(v,str) and len(v.strip())>=MIN_CLAUSE_LENGTH}

# Obligations check

async def _classify_for_obligations(text: str) -> str:
    system = "You are a legal document classifier. Return ONLY one word — no explanation."
    user = (
        "Classify this document into exactly ONE of:\n"
        "nda | rent_agreement | employment_contract | legal_notice | general_contract | other\n\n"
        f"Document excerpt:\n\"\"\"{text[:1500]}\"\"\""
    )
    raw, _, _ = await run_llm_with_tokens(user, system)
    raw = raw.lower().strip()
    if "nda" in raw:                          return "nda"
    if "rent" in raw:                         return "rent_agreement"
    if "employment" in raw:                   return "employment_contract"
    if "notice" in raw or "summons" in raw:   return "legal_notice"
    if "contract" in raw:                     return "general_contract"
    return "other"


_OBL_SYSTEM = (
    "You are an expert legal obligation extraction system. "
    "Extract ALL obligations from the document provided. "
    "Return ONLY valid json — no markdown fences, no text outside the json object.\n\n"
    "For EACH obligation extract these fields:\n"
    "  obligation_id    : POS-001 for must_do, NEG-001 for must_not_do (increment per type)\n"
    "  type             : must_do or must_not_do\n"
    "  party_responsible: who owes the obligation\n"
    "  counterparty     : to whom the obligation is owed\n"
    "  action           : single verb describing the action\n"
    "  object           : what the action applies to\n"
    "  obligation_text  : clear, complete sentence\n"
    "  deadline         : text description or null\n"
    "  normalized_deadline: YYYY-MM-DD or null\n"
    '  schedule         : {"is_recurring": bool, "frequency": "daily/weekly/monthly/yearly/null"}\n'
    "  consequence      : what happens if not met, or null\n"
    "  priority         : High, Medium, or Low\n"
    "  confidence       : 0.0 to 1.0\n"
    "  source_text      : exact sentence from the document\n\n"
    "Return this json structure:\n"
    '{"parties_identified": ["Party A", "Party B"], "obligations": [...]}'
)

_OBL_TYPE_FOCUS = {
    "nda":                 "Focus on: confidentiality, non-disclosure, permitted use, data return, duration.",
    "rent_agreement":      "Focus on: rent payment, maintenance, restrictions, penalties, landlord duties.",
    "employment_contract": "Focus on: duties, confidentiality, notice period, non-compete.",
    "legal_notice":        "Focus on: deadlines, mandatory actions, penalties, legal consequences.",
    "general_contract":    "Focus on: all contractual obligations across parties.",
    "other":               "Extract any obligation-like statement.",
}


async def _extract_obligations(text: str, doc_type: str) -> tuple[list, int, int]:
    focus = _OBL_TYPE_FOCUS.get(doc_type, _OBL_TYPE_FOCUS["other"])
    user_msg = (
        f"{focus}\n"
        "Be exhaustive — do NOT miss implicit obligations. "
        "Split multiple obligations in one sentence into separate entries.\n\n"
        f"Document:\n---\n{text[:12000]}\n---"
    )

    raw, in_tok, out_tok = await run_llm_raw_json(_OBL_SYSTEM, user_msg)
    parsed = extract_json_from_text(raw)
    obligations = parsed.get("obligations", []) if parsed else []

    if len(obligations) < 3:
        logger.warning(f"[obligations] weak extraction ({len(obligations)}) — retrying")
        retry_user = user_msg.replace(
            "Be exhaustive — do NOT miss implicit obligations.",
            "Be EXTREMELY exhaustive — do NOT miss ANY obligation, explicit or implied.",
        )
        raw2, in2, out2 = await run_llm_raw_json(_OBL_SYSTEM, retry_user)
        in_tok += in2; out_tok += out2
        parsed2 = extract_json_from_text(raw2)
        obligations2 = parsed2.get("obligations", []) if parsed2 else []
        if len(obligations2) > len(obligations):
            obligations = obligations2

    logger.info(f"[obligations] extracted {len(obligations)} obligations (doc_type={doc_type})")
    return obligations, in_tok, out_tok

def _diff_obligations(obls1: list, obls2: list) -> dict:
    """
    Diff two obligation lists.
    Matches by normalized action + object similarity.
    Returns structured diff with added, removed, modified, unchanged counts.
    """
    def _obl_key(o):
        action = (o.get("action") or "").lower().strip()
        obj    = (o.get("object") or "").lower().strip()
        party  = (o.get("party_responsible") or "").lower().strip()
        return f"{party}|{action}|{obj}"

    map1 = {_obl_key(o): o for o in obls1}
    map2 = {_obl_key(o): o for o in obls2}

    added    = [o for k, o in map2.items() if k not in map1]
    removed  = [o for k, o in map1.items() if k not in map2]
    modified = []
    unchanged_count = 0

    for k, o1 in map1.items():
        if k not in map2:
            continue
        o2 = map2[k]
        text_sim = SequenceMatcher(
            None,
            (o1.get("obligation_text") or "").lower(),
            (o2.get("obligation_text") or "").lower(),
        ).ratio()
        if text_sim < 0.95:
            modified.append({
                "party_responsible": o1.get("party_responsible"),
                "action":            o1.get("action"),
                "original_text":     o1.get("obligation_text"),
                "revised_text":      o2.get("obligation_text"),
                "original_deadline": o1.get("normalized_deadline"),
                "revised_deadline":  o2.get("normalized_deadline"),
                "severity": "high" if o1.get("priority") == "High" else
                            "medium" if o1.get("priority") == "Medium" else "low",
            })
        else:
            unchanged_count += 1

    # Party-level summary
    def _by_party(obls):
        m = {}
        for o in obls:
            p = o.get("party_responsible", "Unknown")
            m.setdefault(p, 0)
            m[p] += 1
        return m

    return {
        "added":           [{"party_responsible": o.get("party_responsible"),
                             "obligation_text":   o.get("obligation_text"),
                             "priority":          o.get("priority"),
                             "deadline":          o.get("normalized_deadline")} for o in added],
        "removed":         [{"party_responsible": o.get("party_responsible"),
                             "obligation_text":   o.get("obligation_text"),
                             "priority":          o.get("priority")} for o in removed],
        "modified":        modified,
        "summary": {
            "total_added":     len(added),
            "total_removed":   len(removed),
            "total_modified":  len(modified),
            "total_unchanged": unchanged_count,
        },
        "obligations_by_party_doc1": _by_party(obls1),
        "obligations_by_party_doc2": _by_party(obls2),
    }


def _diff_deadlines(obls1: list, obls2: list) -> list:
    """
    Find obligations where the deadline changed between versions.
    Only surfaces cases where both versions have a normalized_deadline.
    """
    def _key(o):
        action = (o.get("action") or "").lower().strip()
        party  = (o.get("party_responsible") or "").lower().strip()
        return f"{party}|{action}"

    map1 = {_key(o): o for o in obls1 if o.get("normalized_deadline")}
    map2 = {_key(o): o for o in obls2 if o.get("normalized_deadline")}

    changes = []
    for k, o1 in map1.items():
        if k not in map2:
            continue
        d1, d2 = o1["normalized_deadline"], map2[k]["normalized_deadline"]
        if d1 != d2:
            changes.append({
                "party_responsible": o1.get("party_responsible"),
                "action":            o1.get("action"),
                "original_deadline": d1,
                "revised_deadline":  d2,
                "direction": "tightened" if d2 < d1 else "extended",
                "obligation_text":   o1.get("obligation_text"),
            })
    return changes

# ---------------------------------------------------------------------------
# Clause matching
# ---------------------------------------------------------------------------

def _match_clauses(c1, c2):
    matched = {}
    for k1,v1 in c1.items():
        if k1 in c2: matched[k1]=k1; continue
        best_k,best_s = None,0.0
        for k2,v2 in c2.items():
            s = _csim(k1,v1,k2,v2)
            if s>best_s: best_s,best_k=s,k2
        matched[k1] = best_k if best_s>=MATCH_THRESHOLD else None
    return matched


# ---------------------------------------------------------------------------
# Text diff stats
# ---------------------------------------------------------------------------

def _compute_text_diff_stats(t1, t2):
    l1,l2 = t1.splitlines(),t2.splitlines()
    sm = SequenceMatcher(None,l1,l2)
    added=removed=changed=unchanged=0
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag=="equal":   unchanged+=i2-i1
        elif tag=="insert": added+=j2-j1
        elif tag=="delete": removed+=i2-i1
        elif tag=="replace":
            nr,na=i2-i1,j2-j1; ch=min(nr,na)
            changed+=ch; removed+=nr-ch; added+=na-ch
    return {
        "lines_added":added,
        "lines_removed":removed,
        "lines_changed":changed,
        "lines_unchanged":unchanged,
        "similarity_score":round(sm.ratio(),4),
        "similarity_percent":f"{round(sm.ratio()*100,1)}%",
    }


# ---------------------------------------------------------------------------
# Clause comparison (produces raw changes list with original/revised values)
# ---------------------------------------------------------------------------

def _compare_clauses(c1, c2):
    matches = _match_clauses(c1,c2)
    changes = []
    for k1,k2 in matches.items():
        v1=c1.get(k1); v2=c2.get(k2) if k2 else None
        if v1 and v2 and _sim(v1,v2)>0.87: continue  # 87% similar → treat as unchanged
        ct = "added" if not v1 and v2 else "removed" if v1 and not v2 else "modified"
        changes.append({"clause":k1,"change_type":ct,"original_value":v1,"revised_value":v2,"severity":_sev(k1),"summary":""})
    mc2 = set(v for v in matches.values() if v)
    for k2,v2 in c2.items():
        if k2 not in mc2 and k2 not in c1:
            changes.append({"clause":k2,"change_type":"added","original_value":None,"revised_value":v2,"severity":_sev(k2),"summary":""})
    return changes


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------

def _risk_score(changes):
    h=sum(1 for c in changes if c["severity"]=="high")
    m=sum(1 for c in changes if c["severity"]=="medium")
    l=sum(1 for c in changes if c["severity"]=="low")
    score=min((h*30)+(m*15)+(l*5),100)
    return {"risk_score":score,"overall_risk_level":"high" if score>=70 else "medium" if score>=30 else "low","high_risk_changes":h}


# ---------------------------------------------------------------------------
# LLM enrichment — per-clause summaries + semantic insights + recommendation
# ---------------------------------------------------------------------------

# System message for enrichment — role persona + output schema.
# Kept here so the user message (_build_llm_prompt) contains only data.
_ENRICHMENT_SYSTEM = (
    "You are a senior legal analyst specialising in contract risk review and negotiation advisory.\n\n"
    "Return ONLY the following JSON object — no markdown fences, no text outside the JSON:\n\n"
    "{\n"
    '  "clause_summaries": {\n'
    '    "<exact_clause_key>": "<One sentence: what changed and why it matters to the signing party>",\n'
    '    "...one entry per clause listed above..."\n'
    "  },\n"
    '  "semantic_insights": [\n'
    '    "<Specific, quantified insight — name exact values/dates/percentages. E.g. Payment terms tightened from Net 30 to Net 15, doubling cash-flow pressure.>",\n'
    '    "<Another specific insight about a different clause change and its business impact>",\n'
    '    "<Continue for all major changes — minimum 4 insights total>",\n'
    '    "<FINAL insight: explicitly state which party the revised contract favours overall and 2-3 specific reasons>"\n'
    "  ],\n"
    '  "recommendation": "<2-3 sentences of actionable guidance. Name the specific clauses to push back on and what outcome to target. Example: Negotiate to restore the liability cap and reinstate the 30-day termination notice. Do not sign until these are resolved.>"\n'
    "}\n\n"
    "Rules:\n"
    "- clause_summaries key must exactly match the clause key in the change list\n"
    "- semantic_insights: minimum 4 items; each must reference specific clause text or numbers\n"
    "- Last semantic_insight must name the favoured party with evidence\n"
    "- recommendation must name at least 2 specific clauses"
)


def _build_llm_prompt(changes):
    """Returns the USER message — role and output schema are in _ENRICHMENT_SYSTEM."""
    change_lines = []
    for i, c in enumerate(changes[:20], 1):
        orig = (c.get("original_value") or "")[:300].replace("\n", " ")
        rev  = (c.get("revised_value")  or "")[:300].replace("\n", " ")
        change_lines.append(
            f"{i}. CLAUSE: {c['clause'].replace('_',' ').upper()}\n"
            f"   Change: {c['change_type']} | Severity: {c['severity']}\n"
            f"   Original : {orig or '[absent]'}\n"
            f"   Revised  : {rev  or '[absent]'}"
        )
    cb = "\n\n".join(change_lines) or "No changes detected."

    doc1_context = "\n".join(
        f"[{c['clause']}]: {(c.get('original_value') or '')[:200]}"
        for c in changes if c.get("original_value")
    )[:2000]
    doc2_context = "\n".join(
        f"[{c['clause']}]: {(c.get('revised_value') or '')[:200]}"
        for c in changes if c.get("revised_value")
    )[:2000]

    return (
        f"Analyse the clause-level differences between two contract versions.\n\n"
        f"DETECTED CLAUSE CHANGES ({len(changes)} total):\n{cb}\n\n"
        f"ORIGINAL DOCUMENT — key clause values:\n{doc1_context or '[not available]'}\n\n"
        f"REVISED DOCUMENT — key clause values:\n{doc2_context or '[not available]'}"
    )


async def _run_llm_enrichment(changes):
    empty = {"clause_summaries": {}, "semantic_insights": [], "recommendation": ""}
    if not changes:
        return empty

    try:
        user_prompt = _build_llm_prompt(changes)
        # JSON mode: forces the model to output valid JSON (no markdown fences, no prose)
        raw, _, _ = await run_llm_raw_json(_ENRICHMENT_SYSTEM, user_prompt)
        result = extract_json_from_text(raw)

        if result and (result.get("semantic_insights") or result.get("clause_summaries")):
            logger.info(
                f"[comparison] LLM OK summaries={len(result.get('clause_summaries', {}))} "
                f"insights={len(result.get('semantic_insights', []))}"
            )
            return result

        # Retry with a concrete worked example
        logger.warning(f"[comparison] LLM attempt 1 weak (raw={raw[:200]!r}) — retrying")
        ex      = changes[0] if changes else {}
        ex_key  = ex.get("clause", "payment_terms")
        ex_orig = (ex.get("original_value") or "original text")[:80]
        ex_rev  = (ex.get("revised_value")  or "revised text")[:80]

        example_block = (
            "Example of the exact json format required:\n"
            "{\n"
            f'  "clause_summaries": {{"{ex_key}": "{ex_key.replace("_"," ").capitalize()} '
            f'changed from \\"{ex_orig[:40]}...\\" to \\"{ex_rev[:40]}...\\" — increases risk for the recipient."}},\n'
            '  "semantic_insights": [\n'
            f'    "{ex_key.replace("_"," ").capitalize()} changed — '
            f'original: {ex_orig[:60]} / revised: {ex_rev[:60]}. This increases financial exposure.",\n'
            '    "Provide a specific, quantified insight for each other changed clause.",\n'
            '    "Final: state which party benefits overall and why."\n'
            "  ],\n"
            '  "recommendation": "Negotiate to restore [clause 1] and [clause 2] before signing."\n'
            "}\n\n"
            "Now produce the real json analysis for the clauses below:\n\n"
        )
        retry_user = example_block + user_prompt
        raw2, _, _ = await run_llm_raw_json(_ENRICHMENT_SYSTEM, retry_user)
        result2 = extract_json_from_text(raw2)
        if result2:
            logger.info("[comparison] Retry succeeded")
            return result2

        logger.error(f"[comparison] Both enrichment attempts failed. raw2={raw2[:200]!r}")
        return empty

    except Exception as e:
        logger.exception(f"[comparison] Enrichment error: {e}")
        return empty


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def compare_documents(
    text1, text2, clauses1, clauses2,
    *, doc1_filename="document_1.pdf", doc2_filename="document_2.pdf",
    session_id="", log_id=None,
):
    t_start = time.perf_counter()

    if len(text1.split()) < 20 or len(text2.split()) < 20:
        raise ValueError("Documents too short for meaningful comparison.")

    c1 = _clean(clauses1); c2 = _clean(clauses2)
    if not c1 or not c2:
        logger.warning("[comparison] Empty clauses — full-text fallback")
        c1 = c1 or {"full_text": text1[:4000]}
        c2 = c2 or {"full_text": text2[:4000]}

    logger.info(f"[comparison] doc1={list(c1.keys())}  doc2={list(c2.keys())}")

    # ── Step 1: Classify for obligation extraction ──────────────────────────
    doc_type1, doc_type2 = await asyncio.gather(
        _classify_for_obligations(text1),
        _classify_for_obligations(text2),
    )
    logger.info(f"[comparison] obligation doc types: ({doc_type1}, {doc_type2})")

    # ── Step 2: All parallel work ───────────────────────────────────────────
    (raw_changes, text_diff_stats,
     (obls1, obl_in1, obl_out1),
     (obls2, obl_in2, obl_out2)) = await asyncio.gather(
        asyncio.to_thread(_compare_clauses, c1, c2),
        asyncio.to_thread(_compute_text_diff_stats, text1, text2),
        _extract_obligations(text1, doc_type1),
        _extract_obligations(text2, doc_type2),
    )

    obl_tokens = obl_in1 + obl_in2 + obl_out1 + obl_out2
    logger.info(
        f"[comparison] obligations: doc1={len(obls1)} doc2={len(obls2)} "
        f"tokens={obl_tokens}"
    )

    # ── Step 3: Diffs ───────────────────────────────────────────────────────
    obligation_diff  = _diff_obligations(obls1, obls2)
    deadline_changes = _diff_deadlines(obls1, obls2)

    # ── Step 4: Risk scoring + LLM enrichment ──────────────────────────────
    risk     = _risk_score(raw_changes)
    llm_data = await _run_llm_enrichment(raw_changes)

    summaries = llm_data.get("clause_summaries", {})
    insights  = llm_data.get("semantic_insights", [])
    rec       = llm_data.get("recommendation", "")

    clause_changes = []
    for c in raw_changes:
        key = c["clause"]
        summary = summaries.get(key, "").strip() or \
                  f"{key.replace('_',' ').capitalize()} clause has been {c['change_type']}."
        clause_changes.append({
            "clause":         key,
            "change_type":    c["change_type"],
            "original_value": c["original_value"],
            "revised_value":  c["revised_value"],
            "severity":       c["severity"],
            "summary":        summary,
        })

    duration_ms = int((time.perf_counter() - t_start) * 1000)
    logger.info(
        f"[comparison] {duration_ms}ms changes={len(clause_changes)} "
        f"obligations_added={obligation_diff['summary']['total_added']} "
        f"obligations_removed={obligation_diff['summary']['total_removed']} "
        f"risk={risk['overall_risk_level']}"
    )

    return {
        "status":      "success",
        "log_id":      log_id,
        "duration_ms": duration_ms,
        "comparison": {
            "session_id":         session_id,
            "doc1_filename":      doc1_filename,
            "doc2_filename":      doc2_filename,
            "compared_at":        datetime.now(timezone.utc).isoformat(),
            "total_changes":      len(clause_changes),
            "high_risk_changes":  risk["high_risk_changes"],
            "overall_risk_level": risk["overall_risk_level"],
            "recommendation":     rec,
            "clause_changes":     clause_changes,
            "semantic_insights":  insights,
            "text_diff_stats":    text_diff_stats,

            # ── NEW fields ──────────────────────────────────────────────
            "obligation_diff":    obligation_diff,
            "deadline_changes":   deadline_changes,
            "obligation_summary": {
                "doc1_total":   len(obls1),
                "doc2_total":   len(obls2),
                "added":        obligation_diff["summary"]["total_added"],
                "removed":      obligation_diff["summary"]["total_removed"],
                "modified":     obligation_diff["summary"]["total_modified"],
                "by_party_doc1": obligation_diff["obligations_by_party_doc1"],
                "by_party_doc2": obligation_diff["obligations_by_party_doc2"],
            },
        },
    }