import asyncio
import hashlib
import os
import time
import logging
import tiktoken
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

from utils.json_utils import extract_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME_OB = os.environ.get("MODEL_NAME_OB", "gpt-4o")
MODEL_NAME     = os.environ.get("MODEL_NAME", "gpt-5-nano")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set in environment variables.")

# Models that do NOT support temperature parameter
_FIXED_TEMPERATURE_MODELS = {
    "gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini",
    "o1", "o1-mini", "o3-mini", "o3",
}

# Models that use max_completion_tokens instead of max_tokens
_MAX_COMPLETION_TOKENS_MODELS = {
    "gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini",
    "o1", "o1-mini", "o3-mini", "o3",
}

TOKEN_CHUNK_SIZE    = 800000
TOKEN_CHUNK_OVERLAP = 500
MAX_OUTPUT_TOKENS   = 4096
MAP_JSON_RETRY_ATTEMPTS = 2

# ---------------------------------------------------------------------------
# Module-level semaphore — shared across all requests on this worker
# ---------------------------------------------------------------------------
_MAP_CONCURRENCY  = 3
_MAP_SEMAPHORE: asyncio.Semaphore | None = None

# Caps concurrent plain-text LLM calls (risk detection, key clauses, etc.)
# to avoid OpenAI rate-limit throttling when many chunks fire in parallel.
_TEXT_CONCURRENCY = 8
_TEXT_SEMAPHORE: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _MAP_SEMAPHORE
    if _MAP_SEMAPHORE is None:
        _MAP_SEMAPHORE = asyncio.Semaphore(_MAP_CONCURRENCY)
        logger.info(f"[ai_model] MAP semaphore created (concurrency={_MAP_CONCURRENCY})")
    return _MAP_SEMAPHORE


def _get_text_semaphore() -> asyncio.Semaphore:
    global _TEXT_SEMAPHORE
    if _TEXT_SEMAPHORE is None:
        _TEXT_SEMAPHORE = asyncio.Semaphore(_TEXT_CONCURRENCY)
        logger.info(f"[ai_model] TEXT semaphore created (concurrency={_TEXT_CONCURRENCY})")
    return _TEXT_SEMAPHORE


_client   = AsyncOpenAI(api_key=OPENAI_API_KEY)
_encoding = tiktoken.encoding_for_model("gpt-4o")


# ---------------------------------------------------------------------------
# Token-accurate chunking
# ---------------------------------------------------------------------------

def split_by_tokens(text: str) -> list[str]:
    if not text:
        return []
    t0       = time.perf_counter()
    all_ids  = _encoding.encode(text)
    n_tokens = len(all_ids)
    logger.info(
        f"[split_by_tokens] {len(text):,} chars → {n_tokens:,} tokens "
        f"({len(text)/n_tokens:.2f} chars/token) in {time.perf_counter()-t0:.3f}s"
    )
    chunks = []
    start  = 0
    while start < n_tokens:
        end       = min(start + TOKEN_CHUNK_SIZE, n_tokens)
        chunk_ids = all_ids[start:end]
        chunks.append(_encoding.decode(chunk_ids))
        start    += TOKEN_CHUNK_SIZE - TOKEN_CHUNK_OVERLAP
    logger.info(f"[split_by_tokens] → {len(chunks)} chunk(s) of ≤{TOKEN_CHUNK_SIZE} tokens")
    return chunks


# ---------------------------------------------------------------------------
# Prompts for PDF analysis (used by generate_analysis / generate_analysis_stream)
# These prompts contain the word "json" — required by OpenAI when
# response_format=json_object is set.
# ---------------------------------------------------------------------------

_MAP_SYSTEM = (
    "You are a document analysis assistant. Analyse the document excerpt provided. "
    "Output ONLY a JSON object with exactly these fields:\n"
    '{"overview":"what this document is — type, subject, and purpose",'
    '"summary":"cover all key points in this excerpt — as long or short as the content requires",'
    '"highlights":["specific fact with number/name/date","fact","fact"]}\n'
    "CRITICAL: overview and summary MUST be plain strings. "
    "highlights MUST be a flat array of strings. "
    "NEVER use nested objects or nested arrays."
)

_MAP_RETRY_SYSTEM = (
    "Output ONLY this JSON object. Nothing else.\n"
    "overview and summary must be plain strings. highlights must be a flat array of strings.\n"
    '{"overview":"...","summary":"...","highlights":["...","...","..."]}'
)

_SYNTH_SYSTEM = (
    "You are given summaries of consecutive sections of a single document. "
    "Write a JSON object with exactly two fields:\n"
    '{"overview":"describe what the entire document is — its type, subject, and main purpose",'
    '"summary":"cover ALL major topics across the entire document. Work through from start to end. '
    "Be specific — include key subjects, people, figures, decisions, and conclusions. "
    'Do NOT repeat topics. Write as much as needed to accurately represent the full document."}\n'
    "CRITICAL: both fields must be plain strings — no nested objects, no arrays. "
    "No prose outside the JSON."
)


def _build_map_messages(text: str, retry: bool = False) -> list[dict]:
    system = _MAP_RETRY_SYSTEM if retry else _MAP_SYSTEM
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Document:\n---\n{text}\n---"},
    ]


def _build_synth_messages(results: list[dict]) -> list[dict]:
    parts = []
    for i, r in enumerate(results, 1):
        overview = r.get("overview", "").strip()
        summary  = r.get("summary",  "").strip()
        if overview or summary:
            lines = [f"[Section {i}]"]
            if overview:
                lines.append(f"Type: {overview}")
            if summary:
                lines.append(f"Content: {summary}")
            parts.append("\n".join(lines))
    body = "\n\n".join(parts)
    return [
        {"role": "system", "content": _SYNTH_SYSTEM},
        {"role": "user",   "content": f"Section summaries (in document order):\n---\n{body}\n---"},
    ]


def _merge_highlights(results: list[dict]) -> list[str]:
    seen:       set  = set()
    highlights: list = []
    for r in results:
        for h in r.get("highlights", []):
            h = h.strip()
            if not h:
                continue
            key = " ".join(h.lower().split())
            if key not in seen:
                seen.add(key)
                highlights.append(h)
    logger.info(f"[_merge_highlights] {len(highlights)} distinct highlight(s) preserved")
    return highlights


# ---------------------------------------------------------------------------
# _build_api_kwargs — shared helper to build model-aware API kwargs
# ---------------------------------------------------------------------------

def _messages_seed(messages: list[dict]) -> int:
    """
    Derives a stable integer seed from the full message content.
    Same document + same prompt → same seed → deterministic LLM output.
    """
    content = "".join(m.get("content", "") for m in messages)
    digest  = hashlib.sha256(content.encode()).hexdigest()
    return int(digest[:8], 16)


def _build_api_kwargs(
    messages:          list[dict],
    use_json:          bool = False,
    streaming:         bool = False,
    max_output_tokens: int  = MAX_OUTPUT_TOKENS,
    model:             str | None = None,
) -> dict:
    """
    Build OpenAI API kwargs handling model differences.

    use_json=True  → sets response_format=json_object
                     ONLY use when messages contain the word "json"
                     (OpenAI requirement) — PDF analysis calls only.

    use_json=False → NO response_format
                     Required for key-clause-extraction, risk-detection,
                     and any plain-text call where prompt lacks "json".

    model          → override the global MODEL_NAME for this call only.
    """
    model = model or os.environ.get("MODEL_NAME", MODEL_NAME)

    kwargs: dict = {
        "model":    model,
        "messages": messages,
    }

    # temperature: not supported by gpt-5-nano, o1, o3 family
    if model not in _FIXED_TEMPERATURE_MODELS:
        kwargs["temperature"] = 0.0

    # seed derived from message content — same input always produces same output
    kwargs["seed"] = _messages_seed(messages)

    # token limit parameter name differs by model
    if model in _MAX_COMPLETION_TOKENS_MODELS:
        kwargs["max_completion_tokens"] = max_output_tokens
    else:
        kwargs["max_tokens"] = max_output_tokens

    # JSON mode — ONLY when prompt contains word "json"
    if use_json:
        kwargs["response_format"] = {"type": "json_object"}

    # streaming
    if streaming:
        kwargs["stream"]         = True
        kwargs["stream_options"] = {"include_usage": True}

    return kwargs

def _build_api_kwargs_ob(
    messages:   list[dict],
    use_json:   bool = False,
    streaming:  bool = False,
) -> dict:
    """
    Build OpenAI API kwargs handling model differences.

    use_json=True  → sets response_format=json_object
                     ONLY use when messages contain the word "json"
                     (OpenAI requirement) — PDF analysis calls only.

    use_json=False → NO response_format
                     Required for key-clause-extraction, risk-detection,
                     and any plain-text call where prompt lacks "json".
    """
    model = os.environ.get("MODEL_NAME_OB", MODEL_NAME_OB)

    kwargs: dict = {
        "model":    model,
        "messages": messages,
    }

    # temperature: not supported by gpt-5-nano, o1, o3 family
    if model not in _FIXED_TEMPERATURE_MODELS:
        kwargs["temperature"] = 0.0

    # seed: makes output deterministic across repeated calls with the same input
    kwargs["seed"] = 42

    # token limit parameter name differs by model
    if model in _MAX_COMPLETION_TOKENS_MODELS:
        kwargs["max_completion_tokens"] = MAX_OUTPUT_TOKENS
    else:
        kwargs["max_tokens"] = MAX_OUTPUT_TOKENS

    # JSON mode — ONLY when prompt contains word "json"
    if use_json:
        kwargs["response_format"] = {"type": "json_object"}

    # streaming
    if streaming:
        kwargs["stream"]         = True
        kwargs["stream_options"] = {"include_usage": True}

    return kwargs


# ---------------------------------------------------------------------------
# _run_inference_json
#
# For: PDF analysis map + synthesis calls
# Routes: POST /analyze, POST /analyze/stream
#
# Sets response_format=json_object.
# Safe because _MAP_SYSTEM and _SYNTH_SYSTEM both contain the word "json".
# ---------------------------------------------------------------------------

async def _run_inference_json(
    messages: list[dict],
    label:    str = "",
) -> tuple[str, int, int]:
    """
    OpenAI call with response_format=json_object.
    Use ONLY when system prompt contains the word 'json'.
    Returns (content, input_tokens, output_tokens).
    """
    tag    = f"[{label}] " if label else ""
    t0     = time.perf_counter()
    kwargs = _build_api_kwargs(messages, use_json=True, streaming=False)

    try:
        response      = await _client.chat.completions.create(**kwargs)
        elapsed       = time.perf_counter() - t0
        content       = response.choices[0].message.content or ""
        input_tokens  = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        logger.info(f"{tag}in={input_tokens} out={output_tokens} in {elapsed:.2f}s")
        return content, input_tokens, output_tokens
    except Exception as e:
        logger.exception(f"{tag}OpenAI API call failed: {e}")
        raise


# ---------------------------------------------------------------------------
# _run_inference_text
#
# For: key-clause-extraction, risk-detection, intent classification
# Routes: POST /key-clause-extraction, POST /detect-risks
#
# NO response_format — plain text output.
# Required because those prompts do NOT contain the word "json" and
# setting json_object would cause a 400 error from OpenAI.
# ---------------------------------------------------------------------------

async def _run_inference_text(
    messages:          list[dict],
    label:             str      = "",
    max_output_tokens: int      = MAX_OUTPUT_TOKENS,
    model:             str | None = None,
) -> tuple[str, int, int]:
    """
    OpenAI call WITHOUT response_format — plain text output.
    Use for key-clause-extraction, risk-detection, and any call
    where the prompt does NOT explicitly ask for JSON.
    Returns (content, input_tokens, output_tokens).
    """
    tag    = f"[{label}] " if label else ""
    t0     = time.perf_counter()
    kwargs = _build_api_kwargs(messages, use_json=False, streaming=False,
                               max_output_tokens=max_output_tokens, model=model)

    async with _get_text_semaphore():
        try:
            response      = await _client.chat.completions.create(**kwargs)
            elapsed       = time.perf_counter() - t0
            content       = response.choices[0].message.content or ""
            input_tokens  = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            logger.info(f"{tag}in={input_tokens} out={output_tokens} in {elapsed:.2f}s")
            return content, input_tokens, output_tokens
        except Exception as e:
            logger.exception(f"{tag}OpenAI API call failed: {e}")
            raise

async def _run_inference_text_obligation(
    messages: list[dict],
    label:    str = "",
) -> tuple[str, int, int]:
    """
    OpenAI call WITHOUT response_format — plain text output.
    Use for key-clause-extraction, risk-detection, and any call
    where the prompt does NOT explicitly ask for JSON.
    Returns (content, input_tokens, output_tokens).
    """
    tag    = f"[{label}] " if label else ""
    t0     = time.perf_counter()
    kwargs = _build_api_kwargs_ob(messages, use_json=False, streaming=False)

    try:
        response      = await _client.chat.completions.create(**kwargs)
        elapsed       = time.perf_counter() - t0
        content       = response.choices[0].message.content or ""
        input_tokens  = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        logger.info(f"{tag}in={input_tokens} out={output_tokens} in {elapsed:.2f}s")
        return content, input_tokens, output_tokens
    except Exception as e:
        logger.exception(f"{tag}OpenAI API call failed: {e}")
        raise



# ---------------------------------------------------------------------------
# _run_inference_stream
#
# For: generate_analysis_stream (streaming SSE)
# Route: POST /analyze/stream
#
# No response_format (incompatible with stream=True on some SDK versions).
# Yields: ("delta", str) | ("done", (input_tokens, output_tokens))
# ---------------------------------------------------------------------------

async def _run_inference_stream(messages: list[dict], label: str = ""):
    """
    Streaming OpenAI call — yields token deltas then final token counts.
    No response_format (not needed; extract_json handles parsing).
    """
    tag    = f"[{label}] " if label else ""
    t0     = time.perf_counter()
    kwargs = _build_api_kwargs(messages, use_json=False, streaming=True)

    input_tokens  = 0
    output_tokens = 0

    try:
        async with await _client.chat.completions.create(**kwargs) as stream:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield ("delta", chunk.choices[0].delta.content)
                if chunk.usage:
                    input_tokens  = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens

        logger.info(
            f"{tag}stream done — in={input_tokens} out={output_tokens} "
            f"({time.perf_counter()-t0:.2f}s)"
        )
    except Exception as e:
        logger.exception(f"{tag}streaming OpenAI call failed: {e}")
        raise

    yield ("done", (input_tokens, output_tokens))


# ---------------------------------------------------------------------------
# run_llm — generic plain-text runner
#
# Routes: POST /key-clause-extraction, POST /detect-risks
#
# Uses _run_inference_TEXT (no response_format) because:
#   - key-clause + risk-detection prompts do NOT contain the word "json"
#   - setting json_object without "json" in messages → 400 error from OpenAI
# ---------------------------------------------------------------------------

async def transcribe_audio(audio_bytes: bytes, filename: str) -> tuple[str, str]:
    """
    Transcribe audio using OpenAI Whisper.
    Accepts any format supported by the Whisper API
    (flac, m4a, mp3, mp4, mpeg, mpga, oga, ogg, wav, webm).

    Uses the transcriptions endpoint (NOT translations) so Whisper transcribes
    in the same language it detects — audio in Hindi returns Hindi text,
    audio in Spanish returns Spanish text, etc.

    response_format="verbose_json" is used to get the detected language back
    alongside the transcript text.

    Returns: (transcribed_text, detected_language_code)
    e.g. ("Hola mundo", "spanish")
    """
    import io
    t0 = time.perf_counter()
    try:
        audio_file = (filename, io.BytesIO(audio_bytes))
        response = await _client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
        )
        elapsed = time.perf_counter() - t0
        detected_language = getattr(response, "language", "unknown")
        logger.info(
            f"[transcribe_audio] done in {elapsed:.2f}s — "
            f"{len(response.text)} chars — language={detected_language}"
        )
        return response.text, detected_language
    except Exception as e:
        logger.exception(f"[transcribe_audio] Whisper API call failed: {e}")
        raise


async def run_llm(
    text:              str,
    system_prompt:     str,
    max_input_tokens:  int      = 50000,
    max_output_tokens: int      = MAX_OUTPUT_TOKENS,
    model:             str | None = None,
) -> str:
    """
    Generic plain-text LLM runner.
    Used by /key-clause-extraction and /detect-risks.
    Does NOT set response_format — plain text output only.
    Pass model to override the global MODEL_NAME for this call only.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Document:\n----------------\n{text}\n----------------"},
    ]
    content, _, _ = await _run_inference_text(messages, "run_llm",
                                              max_output_tokens=max_output_tokens,
                                              model=model)
    return content

async def run_llm_with_tokens(
    text: str,
    system_prompt: str,
    max_input_tokens: int = 50000,
) -> tuple[str, int, int]:
    """
    Same as run_llm() but returns (content, input_tokens, output_tokens).
    Used by obligation detection inside document_comparison.py.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Document:\n----------------\n{text}\n----------------"},
    ]
    content, in_tok, out_tok = await _run_inference_text(messages, "run_llm_with_tokens")
    return content, in_tok, out_tok


async def run_llm_raw(system: str, user: str) -> str:
    """
    LLM call with explicit system and user messages — no document wrapper.
    Use when the caller already embeds all context in the user turn
    (e.g. enrichment prompts for document comparison).
    Does NOT set response_format — plain text output only.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    content, _, _ = await _run_inference_text(messages, "run_llm_raw")
    return content


async def run_llm_raw_json(system: str, user: str) -> tuple[str, int, int]:
    """
    LLM call with explicit system and user messages, forcing JSON output mode.
    Use for structured extraction that requires a guaranteed JSON response
    (e.g. enrichment, obligation extraction in document comparison).

    REQUIREMENT: the combined system+user content MUST contain the word 'json'
    (OpenAI constraint for response_format=json_object).
    Returns (content, input_tokens, output_tokens).
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    content, in_tok, out_tok = await _run_inference_json(messages, "run_llm_raw_json")
    return content, in_tok, out_tok


# ---------------------------------------------------------------------------
# _run_map_chunk — one map chunk with semaphore + retry
# Uses _run_inference_json (PDF analysis prompts contain "json")
# ---------------------------------------------------------------------------

async def _run_map_chunk(i: int, total: int, chunk_text: str) -> tuple[dict, int, int]:
    _EMPTY = {"overview": "", "summary": "", "highlights": []}
    sem    = _get_semaphore()
    lbl    = f"map {i+1}/{total}"

    async with sem:
        t_chunk    = time.perf_counter()
        parsed     = None
        in_tokens  = 0
        out_tokens = 0

        for attempt in range(1, MAP_JSON_RETRY_ATTEMPTS + 1):
            try:
                messages = _build_map_messages(chunk_text, retry=(attempt > 1))
                if attempt > 1:
                    logger.warning(f"[generate_analysis] [{lbl}] retry {attempt}")
                raw, i_tok, o_tok = await _run_inference_json(
                    messages, f"{lbl}-a{attempt}"
                )
                in_tokens  += i_tok
                out_tokens += o_tok
            except Exception as e:
                logger.error(f"[generate_analysis] [{lbl}] inference error: {e}")
                break

            candidate = extract_json(raw)
            if candidate.get("overview") or candidate.get("highlights"):
                parsed = candidate
                if attempt > 1:
                    logger.info(f"[generate_analysis] [{lbl}] retry {attempt} produced valid JSON")
                break
            else:
                logger.warning(
                    f"[generate_analysis] [{lbl}] attempt {attempt}: no usable JSON — "
                    + ("retrying" if attempt < MAP_JSON_RETRY_ATTEMPTS else "giving up")
                )

        if not isinstance(parsed, dict) or not (parsed.get("overview") or parsed.get("highlights")):
            parsed = dict(_EMPTY)

        logger.info(
            f"[generate_analysis] [{lbl}] done ({time.perf_counter()-t_chunk:.2f}s) "
            f"overview={'yes' if parsed.get('overview') else 'empty'} "
            f"highlights={len(parsed.get('highlights', []))}"
        )
        return parsed, in_tokens, out_tokens


# ---------------------------------------------------------------------------
# generate_analysis
# Route: POST /analyze
# ---------------------------------------------------------------------------

async def generate_analysis(merged_text: str) -> tuple[dict, int, int]:
    _EMPTY = {"overview": "", "summary": "", "highlights": []}
    if not merged_text or not merged_text.strip():
        return dict(_EMPTY), 0, 0

    t0     = time.perf_counter()
    chunks = split_by_tokens(merged_text)
    logger.info(f"[generate_analysis] {len(chunks)} chunk(s) in {time.perf_counter()-t0:.3f}s")

    t_pipeline    = time.perf_counter()
    total_in_tok  = 0
    total_out_tok = 0

    logger.info(
        f"[generate_analysis] MAP: {len(chunks)} chunk(s) "
        f"(parallel, concurrency={_MAP_CONCURRENCY})"
    )
    map_tasks   = [_run_map_chunk(i, len(chunks), ct) for i, ct in enumerate(chunks)]
    raw_results = list(await asyncio.gather(*map_tasks))

    map_results = []
    for parsed, i_tok, o_tok in raw_results:
        map_results.append(parsed)
        total_in_tok  += i_tok
        total_out_tok += o_tok

    valid_count = sum(1 for r in map_results if r.get("overview") or r.get("highlights"))
    logger.info(f"[generate_analysis] MAP complete — {len(map_results)} total, {valid_count} with content")

    if len(map_results) == 1:
        logger.info("[generate_analysis] single chunk — returning directly")
        result = map_results[0]

    else:
        all_highlights = _merge_highlights(map_results)

        logger.info(f"[generate_analysis] SYNTHESIS: overview+summary from {valid_count} chunk(s)")
        t_synth      = time.perf_counter()
        synth_result = {"overview": "", "summary": ""}
        try:
            synth_messages = _build_synth_messages(
                [r for r in map_results if r.get("overview") or r.get("summary")]
            )
            raw_synth, s_in, s_out = await _run_inference_json(synth_messages, "synthesis")
            total_in_tok  += s_in
            total_out_tok += s_out

            parsed_synth = extract_json(raw_synth)
            ov = parsed_synth.get("overview", "")
            sm = parsed_synth.get("summary",  "")
            synth_result["overview"] = (ov if isinstance(ov, str) else str(ov)).strip()
            synth_result["summary"]  = (sm if isinstance(sm, str) else str(sm)).strip()

            logger.info(
                f"[generate_analysis] SYNTHESIS done ({time.perf_counter()-t_synth:.2f}s) "
                f"overview={'yes' if synth_result['overview'] else 'empty'} "
                f"summary={'yes' if synth_result['summary'] else 'empty'}"
            )
        except Exception as e:
            logger.error(f"[generate_analysis] SYNTHESIS failed: {e} — fallback to first chunk")

        if not synth_result["overview"]:
            synth_result["overview"] = next(
                (r["overview"] for r in map_results if r.get("overview")), ""
            )
        if not synth_result["summary"]:
            synth_result["summary"]  = next(
                (r["summary"]  for r in map_results if r.get("summary")),  ""
            )

        result = {
            "overview":   synth_result["overview"],
            "summary":    synth_result["summary"],
            "highlights": all_highlights,
        }

    if not isinstance(result, dict):
        result = dict(_EMPTY)

    logger.info(
        f"[generate_analysis] total={time.perf_counter()-t_pipeline:.2f}s "
        f"tokens={total_in_tok}in/{total_out_tok}out "
        f"highlights={len(result.get('highlights', []))}"
    )
    return result, total_in_tok, total_out_tok


# ---------------------------------------------------------------------------
# _run_map_chunk_stream — stream one map chunk, yield deltas then result
# Uses _run_inference_stream (no response_format, streaming=True)
# ---------------------------------------------------------------------------

async def _run_map_chunk_stream(i: int, total: int, chunk_text: str):
    _EMPTY = {"overview": "", "summary": "", "highlights": []}
    sem    = _get_semaphore()
    lbl    = f"map {i+1}/{total}"

    async with sem:
        t_chunk    = time.perf_counter()
        in_tokens  = 0
        out_tokens = 0
        parsed     = None

        for attempt in range(1, MAP_JSON_RETRY_ATTEMPTS + 1):
            messages   = _build_map_messages(chunk_text, retry=(attempt > 1))
            raw_buffer = ""

            if attempt > 1:
                logger.warning(f"[stream] [{lbl}] retry {attempt}")

            try:
                async for event_type, payload in _run_inference_stream(
                    messages, f"{lbl}-a{attempt}"
                ):
                    if event_type == "delta":
                        raw_buffer += payload
                        yield ("delta", payload)
                    elif event_type == "done":
                        in_tokens  += payload[0]
                        out_tokens += payload[1]
            except Exception as e:
                logger.error(f"[stream] [{lbl}] inference error: {e}")
                break

            candidate = extract_json(raw_buffer)
            if candidate.get("overview") or candidate.get("highlights"):
                parsed = candidate
                break
            else:
                logger.warning(
                    f"[stream] [{lbl}] attempt {attempt}: no usable JSON — "
                    + ("retrying" if attempt < MAP_JSON_RETRY_ATTEMPTS else "giving up")
                )

        if not isinstance(parsed, dict) or not (parsed.get("overview") or parsed.get("highlights")):
            parsed = dict(_EMPTY)

        logger.info(
            f"[stream] [{lbl}] done ({time.perf_counter()-t_chunk:.2f}s) "
            f"overview={'yes' if parsed.get('overview') else 'empty'} "
            f"highlights={len(parsed.get('highlights', []))}"
        )
        yield ("chunk_done", {**parsed, "_in_tokens": in_tokens, "_out_tokens": out_tokens})


# ---------------------------------------------------------------------------
# generate_analysis_stream
# Route: POST /analyze/stream
# ---------------------------------------------------------------------------

async def generate_analysis_stream(merged_text: str):
    """
    Sequential SSE streaming — each chunk streams tokens live to client.

    Yields (event_type, payload):
      ("chunk_start",    {"chunk": N, "total": N})
      ("token",          {"chunk": N, "delta": str})
      ("chunk_done",     {"chunk": N, "total": N, "overview": str,
                          "new_highlights": [...], "all_highlights_so_far": [...]})
      ("synthesis_start", {})
      ("token",          {"chunk": "synthesis", "delta": str})
      ("synthesis_done", {"overview": str, "summary": str})
      ("done",           {"input_tokens": N, "output_tokens": N, "total_tokens": N})
    """
    if not merged_text or not merged_text.strip():
        yield ("done", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        return

    t0            = time.perf_counter()
    chunks        = split_by_tokens(merged_text)
    total_in_tok  = 0
    total_out_tok = 0
    map_results:    list = []
    all_highlights: list = []
    seen_keys:      set  = set()

    logger.info(f"[generate_analysis_stream] {len(chunks)} chunk(s) — sequential streaming")

    # ── MAP — sequential so tokens reach the client immediately ──────────
    for i, chunk_text in enumerate(chunks):
        yield ("chunk_start", {"chunk": i + 1, "total": len(chunks)})

        parsed = None
        async for event_type, payload in _run_map_chunk_stream(i, len(chunks), chunk_text):
            if event_type == "delta":
                yield ("token", {"chunk": i + 1, "delta": payload})

            elif event_type == "chunk_done":
                parsed         = payload
                total_in_tok  += payload.pop("_in_tokens",  0)
                total_out_tok += payload.pop("_out_tokens", 0)

                new_highlights = []
                for h in parsed.get("highlights", []):
                    h = h.strip()
                    if not h:
                        continue
                    key = " ".join(h.lower().split())
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_highlights.append(h)
                        new_highlights.append(h)

                map_results.append(parsed)

                yield ("chunk_done", {
                    "chunk":                 i + 1,
                    "total":                 len(chunks),
                    "overview":              parsed.get("overview", ""),
                    "new_highlights":        new_highlights,
                    "all_highlights_so_far": list(all_highlights),
                })
                logger.info(
                    f"[stream] [map {i+1}/{len(chunks)}] yielded — "
                    f"overview={'yes' if parsed.get('overview') else 'empty'} "
                    f"new_highlights={len(new_highlights)} total={len(all_highlights)}"
                )

    valid_count = sum(1 for r in map_results if r.get("overview") or r.get("highlights"))
    logger.info(f"[stream] MAP complete — {len(map_results)} total, {valid_count} with content")

    # ── SYNTHESIS ────────────────────────────────────────────────────────
    synth_overview = ""
    synth_summary  = ""

    if len(map_results) == 1:
        synth_overview = map_results[0].get("overview", "")
        synth_summary  = map_results[0].get("summary",  "")
        yield ("synthesis_done", {"overview": synth_overview, "summary": synth_summary})

    else:
        yield ("synthesis_start", {})
        synth_buffer = ""
        try:
            synth_messages = _build_synth_messages(
                [r for r in map_results if r.get("overview") or r.get("summary")]
            )
            async for event_type, payload in _run_inference_stream(synth_messages, "synthesis"):
                if event_type == "delta":
                    synth_buffer += payload
                    yield ("token", {"chunk": "synthesis", "delta": payload})
                elif event_type == "done":
                    total_in_tok  += payload[0]
                    total_out_tok += payload[1]

            parsed_synth = extract_json(synth_buffer)
            ov = parsed_synth.get("overview", "")
            sm = parsed_synth.get("summary",  "")
            synth_overview = (ov if isinstance(ov, str) else str(ov)).strip()
            synth_summary  = (sm if isinstance(sm, str) else str(sm)).strip()
            logger.info(
                f"[stream] SYNTHESIS done — "
                f"overview={'yes' if synth_overview else 'empty'} "
                f"summary={'yes' if synth_summary else 'empty'}"
            )
        except Exception as e:
            logger.error(f"[stream] SYNTHESIS failed: {e} — fallback to first chunk")

        if not synth_overview:
            synth_overview = next(
                (r["overview"] for r in map_results if r.get("overview")), ""
            )
        if not synth_summary:
            synth_summary  = next(
                (r["summary"]  for r in map_results if r.get("summary")),  ""
            )

        yield ("synthesis_done", {"overview": synth_overview, "summary": synth_summary})

    logger.info(
        f"[stream] total={time.perf_counter()-t0:.2f}s "
        f"tokens={total_in_tok}in/{total_out_tok}out "
        f"highlights={len(all_highlights)}"
    )
    yield ("done", {
        "input_tokens":  total_in_tok,
        "output_tokens": total_out_tok,
        "total_tokens":  total_in_tok + total_out_tok,
    })