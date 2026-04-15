"""
s_json_utils.py

All JSON extraction helpers used by s_ai_model.py, s_route.py, and any
other module that needs to parse LLM output.

Moved here so s_ai_model.py does NOT need to import from s_main/s_route,
which caused a circular-import / missing-symbol error after the router
refactor.
"""

import re
import json
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level string fixes
# ---------------------------------------------------------------------------

def _fix_json_string(raw: str) -> str:
    raw = raw.replace("```json", "").replace("```", "").strip()
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)
    return raw


def _extract_field_by_regex(text: str, field: str) -> str:
    pattern = rf'"{field}"\s*:\s*"(.*?)"(?=\s*[,}}])'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_highlights_by_regex(text: str) -> list:
    match = re.search(r'"highlights"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        match_open = re.search(r'"highlights"\s*:\s*\[(.*)', text, re.DOTALL)
        if not match_open:
            return []
        inner = match_open.group(1)
    else:
        inner = match.group(1)
    items = re.findall(r'"((?:[^"\\]|\\.)*)"', inner, re.DOTALL)
    return [i.strip() for i in items if i.strip()]


# ---------------------------------------------------------------------------
# Field flatteners — handle nested dicts/lists the LLM sometimes returns
# ---------------------------------------------------------------------------

def _flatten_field(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [_flatten_field(v) for v in value]
        return ". ".join(p for p in parts if p)
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            flat = _flatten_field(v)
            if flat:
                label = str(k).replace("_", " ").capitalize()
                parts.append(f"{label}: {flat}")
        return ". ".join(parts)
    return str(value).strip()


def _flatten_highlights(value) -> list:
    if not value:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        items = []
        for k, v in value.items():
            flat = _flatten_field(v)
            if flat:
                label = str(k).replace("_", " ").capitalize()
                items.append(f"{label}: {flat}")
        return items
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
            elif isinstance(item, dict):
                flat = _flatten_field(item)
                if flat:
                    result.append(flat)
            elif item is not None:
                s = str(item).strip()
                if s:
                    result.append(s)
        return result
    return []


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _postprocess_highlights(data: dict) -> dict:
    if "highlights" in data and isinstance(data["highlights"], list):
        cleaned = []
        for item in data["highlights"]:
            if isinstance(item, dict):
                text = ""
                for key in ("fact", "highlight", "text", "value", "item", "point"):
                    if isinstance(item.get(key), str):
                        text = item[key]
                        break
                if not text:
                    text = next((v for v in item.values() if isinstance(v, str)), "")
                if text:
                    cleaned.append(text.replace('"', '').strip())
                continue
            if not isinstance(item, str):
                continue
            parts = item.split('", "') if '", "' in item else [item]
            for p in parts:
                cleaned.append(p.replace('"', '').strip())
        data["highlights"] = [h for h in cleaned if h]
    return data


def _normalize_parsed(data, label: str = "") -> dict:
    empty = {"overview": "", "summary": "", "highlights": []}

    if isinstance(data, list):
        if not data:
            return empty
        if isinstance(data[0], dict):
            merged: dict = {"overview": "", "summary": "", "highlights": []}
            for item in data:
                if not isinstance(item, dict):
                    continue
                if not merged["overview"] and item.get("overview"):
                    merged["overview"] = item["overview"]
                if not merged["summary"] and item.get("summary"):
                    merged["summary"] = item["summary"]
                merged["highlights"].extend(item.get("highlights") or [])
            logger.warning(f"extract_json {label}: got list of dicts — merged {len(data)} item(s)")
            data = merged
        elif isinstance(data[0], str):
            return {"overview": "", "summary": "", "highlights": [s for s in data if s]}
        else:
            return empty

    if not isinstance(data, dict):
        logger.error(f"extract_json {label}: unexpected type {type(data).__name__}")
        return empty

    overview = data.get("overview", "")
    if not isinstance(overview, str):
        logger.warning(f"extract_json {label}: overview is {type(overview).__name__} — flattening")
        overview = _flatten_field(overview)

    summary = data.get("summary", "")
    if not isinstance(summary, str):
        logger.warning(f"extract_json {label}: summary is {type(summary).__name__} — flattening")
        summary = _flatten_field(summary)

    raw_highlights = data.get("highlights", [])
    if not isinstance(raw_highlights, list) or (
        raw_highlights and not isinstance(raw_highlights[0], str)
    ):
        logger.warning(f"extract_json {label}: highlights has non-string items — flattening")
        highlights = _flatten_highlights(raw_highlights)
    else:
        highlights = raw_highlights

    return {
        "overview":   overview.strip(),
        "summary":    summary.strip(),
        "highlights": highlights,
    }


def _recover_truncated_json(text: str) -> dict | None:
    result = {"overview": "", "summary": "", "highlights": []}
    ov_match = re.search(r'"overview"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if ov_match:
        result["overview"] = ov_match.group(1).strip()
    sm_match = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if sm_match:
        result["summary"] = sm_match.group(1).strip()
    result["highlights"] = _extract_highlights_by_regex(text)
    if result["overview"] or result["summary"] or result["highlights"]:
        logger.info(
            f"_recover_truncated_json: recovered "
            f"overview={'yes' if result['overview'] else 'no'} "
            f"summary={'yes' if result['summary'] else 'no'} "
            f"highlights={len(result['highlights'])}"
        )
        return result
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _recover_truncated_array(text: str, field: str) -> dict | None:
    """
    Recover a top-level array field from truncated JSON by extracting every
    complete { } object found before the truncation point.
    Used as a last resort when the LLM hits its max_output_tokens limit mid-stream.
    """
    match = re.search(rf'"{re.escape(field)}"\s*:\s*\[', text)
    if not match:
        return None

    array_content = text[match.end():]
    items = []
    depth = 0
    in_string = False
    escape_next = False
    start = None

    for i, ch in enumerate(array_content):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(array_content[start : i + 1])
                    if isinstance(obj, dict):
                        items.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None

    if items:
        logger.warning(
            f"extract_json_raw: truncation recovery found {len(items)} "
            f"complete item(s) in '{field}' array"
        )
        return {field: items}
    return None


def extract_json_raw(text: str) -> dict:
    """
    Extract JSON from LLM output and return it AS-IS — no schema normalization.
    Use this for key-clause-extraction, risk-detection, and any route that
    returns its own field structure (NOT overview/summary/highlights).
    """
    if not text or not text.strip():
        return {}

    cleaned = _fix_json_string(text)

    # Strategy 1 — direct parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Strategy 2 — isolate { } block
    brace_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if brace_match:
        candidate = brace_match.group()
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 3 — truncation recovery: extract complete objects from known array fields
    for field in ("key_clauses", "risks", "flags", "obligations", "clauses"):
        recovered = _recover_truncated_array(cleaned, field)
        if recovered:
            return recovered

    logger.error("extract_json_raw: could not parse JSON — returning empty dict")
    return {}


def extract_json(text: str) -> dict:
    """
    Robustly extract JSON from LLM output.

    Strategy 1 — direct parse after cleaning
    Strategy 2 — isolate { } block + quote-escape repair
    Strategy 3 — truncation recovery
    Strategy 4 — per-field regex (last resort)
    """
    empty = {"overview": "", "summary": "", "highlights": []}
    if not text or not text.strip():
        logger.warning("extract_json: received empty text")
        return empty

    cleaned = _fix_json_string(text)

    try:
        data = json.loads(cleaned)
        logger.debug("extract_json: strategy 1 succeeded")
        return _postprocess_highlights(_normalize_parsed(data, "strategy-1"))
    except json.JSONDecodeError as e:
        logger.debug(f"extract_json: strategy 1 failed — {e}")

    brace_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if brace_match:
        candidate = brace_match.group()
        try:
            data = json.loads(candidate)
            return _postprocess_highlights(_normalize_parsed(data, "strategy-2"))
        except json.JSONDecodeError as e:
            logger.warning(f"extract_json: strategy 2 failed ({e}), trying repair")
            repaired = re.sub(r'(?<=[^\\])"(?=[^,\]}\n:}{\[])', r'\\"', candidate)
            try:
                data = json.loads(repaired)
                return _postprocess_highlights(_normalize_parsed(data, "strategy-2-repair"))
            except json.JSONDecodeError as e2:
                logger.warning(f"extract_json: strategy 2 repair failed — {e2}")
    else:
        logger.warning("extract_json: no brace block found in output")

    recovered = _recover_truncated_json(cleaned)
    if recovered:
        logger.info("extract_json: strategy 3 (truncation recovery) succeeded")
        return _postprocess_highlights(recovered)

    logger.error("extract_json: all strategies failed — falling back to regex")
    overview   = _extract_field_by_regex(cleaned, "overview")
    summary    = _extract_field_by_regex(cleaned, "summary")
    highlights = _extract_highlights_by_regex(cleaned)
    if overview or summary or highlights:
        return {"overview": overview, "summary": summary, "highlights": highlights}

    logger.error("extract_json: all strategies exhausted — returning empty")
    return empty