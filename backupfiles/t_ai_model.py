"""
t_ai_model.py
=============
Mistral-7B-Instruct inference for hierarchical PDF summarization.

Features
--------
- Lazy model loading  (call load_model() once from FastAPI lifespan)
- GPU semaphore       (max MAX_CONCURRENT_GPU simultaneous generate() calls)
- Retry + backoff     (up to MAX_RETRY attempts per call)
- Three prompt modes:
    generate_analysis()   — analyse one text chunk   (Level 0)
    synthesize_section()  — condense N chunk results  (Level 1)
    synthesize_final()    — condense all sections     (Level 2)
- Correct device handling (device chosen before model load; no mismatch)
- Single tokenize pass with truncation (no double-tokenize bug)
- Correct decode slice  (uses truncated input length, not pre-truncation)
- asyncio.get_running_loop() instead of deprecated get_event_loop()
"""

import asyncio
import logging

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME         = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_INPUT_TOKENS   = 6_000   # hard token limit per inference call
MAX_NEW_TOKENS     = 600     # max tokens the model may generate per call
MAX_CONCURRENT_GPU = 3       # GPU semaphore width  (prevents OOM)
MAX_RETRY          = 3       # retry attempts for transient failures

# ---------------------------------------------------------------------------
# Module-level state  (populated by load_model())
# ---------------------------------------------------------------------------
_tokenizer  = None
_model      = None
_device     = None
_gpu_sem: asyncio.Semaphore | None = None   # created lazily on first use


def _get_sem() -> asyncio.Semaphore:
    """Return (and lazily create) the GPU semaphore on the running event loop."""
    global _gpu_sem
    if _gpu_sem is None:
        _gpu_sem = asyncio.Semaphore(MAX_CONCURRENT_GPU)
    return _gpu_sem


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model() -> None:
    """
    Load tokenizer and model weights into memory.

    - Device is chosen BEFORE the model is loaded (no device_map="auto" mismatch).
    - CUDA → float16 with cudnn.benchmark and allow_tf32.
    - CPU  → float32 (float16 not supported on CPU).
    - Safe to call multiple times — subsequent calls are no-ops.
    """
    global _tokenizer, _model, _device

    if _model is not None:
        return  # already loaded

    from transformers import AutoModelForCausalLM, AutoTokenizer

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading {MODEL_NAME} onto {_device} …")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if _device == "cuda":
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16
        ).to(_device)
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32
        )

    _model.eval()
    logger.info("Model ready.")


# ---------------------------------------------------------------------------
# Core synchronous inference  (runs in thread pool via run_in_executor)
# ---------------------------------------------------------------------------

def _run_inference(prompt: str) -> str:
    """
    Tokenize → generate → decode.  Entirely synchronous.

    Correctness fixes vs original code
    -----------------------------------
    - Single tokenize call (was tokenized twice — once to count, once to truncate)
    - input_length taken from the truncated tensor (not pre-truncation encoding)
    - inputs.to(_device) consistent with model device (no device_map mismatch)
    - GPU memory freed in finally block regardless of success or failure
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")

    # Single tokenize pass with truncation
    inputs       = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    input_length = inputs["input_ids"].shape[1]   # length AFTER truncation

    if input_length == MAX_INPUT_TOKENS:
        logger.warning(f"Prompt truncated to {MAX_INPUT_TOKENS} tokens.")

    inputs = inputs.to(_device)

    try:
        with torch.no_grad():
            output = _model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=_tokenizer.eos_token_id,
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "GPU out of memory. Reduce chunk size or MAX_INPUT_TOKENS."
        )
    finally:
        del inputs
        if _device == "cuda":
            torch.cuda.empty_cache()

    # Decode only newly generated tokens (slice from truncated input_length)
    new_tokens = output[0][input_length:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Async wrapper with semaphore + retry
# ---------------------------------------------------------------------------

async def _call_model(prompt: str) -> str:
    """
    Run inference asynchronously with GPU semaphore and retry.

    - Acquires semaphore before submitting to the thread pool.
    - Retries up to MAX_RETRY times with exponential back-off (1 s, 2 s, 4 s).
    - Returns "" on permanent failure (caller handles empty result gracefully).
    """
    loop = asyncio.get_running_loop()
    sem  = _get_sem()

    for attempt in range(MAX_RETRY):
        try:
            async with sem:
                result = await loop.run_in_executor(None, _run_inference, prompt)
            return result
        except RuntimeError as exc:
            if attempt == MAX_RETRY - 1:
                logger.error(
                    f"Inference failed after {MAX_RETRY} attempts: {exc}"
                )
                return ""
            wait = 2 ** attempt   # 1 s → 2 s → 4 s
            logger.warning(
                f"Inference attempt {attempt+1} failed ({exc}); "
                f"retrying in {wait}s …"
            )
            await asyncio.sleep(wait)

    return ""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _chunk_prompt(text: str) -> str:
    """Level 0 prompt: analyse a single content chunk."""
    return f"""<s>[INST]
You are an expert document analyst with deep knowledge across legal, medical,
financial, technical, academic, and business domains.

Read the document EXCERPT below and return ONLY a JSON object with these three fields:

{{
  "overview": "1-2 sentences: what TYPE of document this is, its subject, purpose, and intended audience.",
  "summary": "4-6 sentences covering the most important content in this excerpt.",
  "highlights": [
    "One specific factual sentence containing a real number, date, name, percentage, or clause.",
    "One specific factual sentence containing a real number, date, name, percentage, or clause.",
    "One specific factual sentence containing a real number, date, name, percentage, or clause.",
    "One specific factual sentence containing a real number, date, name, percentage, or clause."
  ]
}}

STRICT RULES:
- Output ONLY the JSON object. No markdown, no backticks, no commentary.
- highlights: 4-6 items. Each must be a complete factual sentence with real values.
- Do NOT write vague statements like "The document contains important information."
- If a section has insufficient information, write "Not enough information available."
- Escape all special characters in strings properly.

Document excerpt:
----------------
{text}
----------------
[/INST]"""


def _section_prompt(chunk_summaries: list[str], section_label: str) -> str:
    """Level 1 prompt: synthesize N chunk summaries into one section summary."""
    numbered = "\n\n".join(
        f"[Chunk {i+1}]\n{s}" for i, s in enumerate(chunk_summaries)
    )
    return f"""<s>[INST]
You are an expert document analyst. You are given {len(chunk_summaries)} sequential
chunk summaries from {section_label} of a large document.

Synthesize them into ONE coherent JSON object:

{{
  "overview": "1-2 sentences describing what this section covers.",
  "summary": "4-6 sentences summarising the key content across all chunks in this section.",
  "highlights": [
    "Most important specific fact, figure, or finding from this section.",
    "Second most important fact with a real number, date, name, or clause.",
    "Third most important fact.",
    "Fourth most important fact."
  ]
}}

STRICT RULES:
- Output ONLY the JSON. No markdown, no backticks.
- Do NOT simply list each chunk — synthesize them into a unified narrative.
- Eliminate any repeated points across chunks.
- highlights: 4-6 items, each a complete factual sentence with real values.

Chunk summaries:
----------------
{numbered}
----------------
[/INST]"""


def _final_prompt(section_summaries: list[str], total_pages: int) -> str:
    """Level 2 prompt: synthesize all section summaries into the final output."""
    numbered = "\n\n".join(
        f"[Section {i+1}]\n{s}" for i, s in enumerate(section_summaries)
    )
    return f"""<s>[INST]
You are an expert document analyst. You have been given {len(section_summaries)} section-level
summaries that together cover a {total_pages}-page document in full.

Produce ONE authoritative document-level JSON object:

{{
  "overview": "1-2 sentences: document type, overall subject, purpose, and intended audience.",
  "summary": "5-7 sentences covering the document's purpose, key arguments or findings,
              important data or obligations, conclusions, and overall significance.",
  "highlights": [
    "The single most critical fact, figure, deadline, or obligation in the entire document.",
    "Second most important fact with a real number, date, name, percentage, or clause.",
    "Third most important fact.",
    "Fourth most important fact.",
    "Fifth most important fact.",
    "Sixth most important fact.",
    "Seventh most important fact (if applicable).",
    "Eighth most important fact (if applicable).",
    "Ninth most important fact (if applicable).",
    "Tenth most important fact (if applicable)."
  ]
}}

STRICT RULES:
- Output ONLY the JSON. No markdown, no backticks, no commentary.
- highlights: 6-10 items ranked by importance. Each must be a complete factual sentence.
- Do NOT repeat information already in the overview or summary.
- Eliminate all repetition across sections.
- Write in a professional, neutral tone.

Section summaries:
----------------
{numbered}
----------------
[/INST]"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_analysis(text: str) -> str:
    """
    Level 0 — analyse a single text chunk.
    Returns the raw model output string (caller parses with extract_json()).
    """
    return await _call_model(_chunk_prompt(text))


async def synthesize_section(
    chunk_results: list[dict],
    section_label: str = "a section",
) -> str:
    """
    Level 1 — condense the parsed results of N chunks into one section summary.

    Parameters
    ----------
    chunk_results : list of dicts with keys overview / summary / highlights
    section_label : human-readable label used in the prompt

    Returns raw model output string (caller parses with extract_json()).
    """
    chunk_summaries = []
    for r in chunk_results:
        parts = []
        if r.get("overview"):
            parts.append(f"Overview: {r['overview']}")
        if r.get("summary"):
            parts.append(f"Summary: {r['summary']}")
        if r.get("highlights"):
            hl = "\n".join(f"  - {h}" for h in r["highlights"])
            parts.append(f"Highlights:\n{hl}")
        chunk_summaries.append("\n".join(parts))

    return await _call_model(_section_prompt(chunk_summaries, section_label))


async def synthesize_final(
    section_results: list[dict],
    total_pages: int,
) -> str:
    """
    Level 2 — condense all section-level summaries into the final document output.

    Parameters
    ----------
    section_results : list of dicts with keys overview / summary / highlights
    total_pages     : original PDF page count (used in the prompt for context)

    Returns raw model output string (caller parses with extract_json()).
    """
    section_summaries = []
    for r in section_results:
        parts = []
        if r.get("overview"):
            parts.append(f"Overview: {r['overview']}")
        if r.get("summary"):
            parts.append(f"Summary: {r['summary']}")
        if r.get("highlights"):
            hl = "\n".join(f"  - {h}" for h in r["highlights"])
            parts.append(f"Highlights:\n{hl}")
        section_summaries.append("\n".join(parts))

    return await _call_model(_final_prompt(section_summaries, total_pages))

async def run_llm(
    text: str,
    system_prompt: str,
    max_input_tokens: int = MAX_INPUT_TOKENS
) -> str:
    """
    Generic reusable LLM runner.
    Accepts:
    - text (document content)
    - system_prompt (custom instruction per use-case)

    Returns:
    - model output (string)
    """

    prompt = f"""<s>[INST]
{system_prompt}

Document:
----------------
{text[:4000]}
----------------
[/INST]
"""

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _run_inference, prompt)

    return result

