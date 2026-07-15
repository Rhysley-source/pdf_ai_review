"""
model_file.py

Local LLM wrapper for microsoft/Phi-4-mini-instruct.

Exposed as a route on the SAME router as the rest of the PDF API
(see routes/route.py: POST /phi/generate) rather than a separate service.

The model is loaded lazily on first call, not at import time — routes/route.py
is imported once at app startup for every request type, so eagerly loading a
multi-GB local model here would slow every worker's startup and load it even
in workers that never receive a Phi request.

CAVEAT: if this app runs with multiple gunicorn workers, each worker that
handles a /phi/generate request loads its own copy of the model into
memory/GPU (same resource-contention pattern the old PaddleOCR-VL setup had
before it was replaced with a remote OCR call). If that's a problem, pin
this app to --workers 1, or move this back out into its own process.

.env (optional):
  PHI_MODEL_ID    default: microsoft/Phi-4-mini-instruct
  PHI_CONCURRENCY default: 1  (max concurrent generate() calls)
"""

import asyncio
import logging
import os
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("PHI_MODEL_ID", "microsoft/Phi-4-mini-instruct")

_model:      AutoModelForCausalLM | None = None
_tokenizer:  AutoTokenizer        | None = None
_load_lock = threading.Lock()

# Serialises generate() calls — a single model instance shouldn't run
# concurrent generations (GPU/CPU contention risk).
_PHI_CONCURRENCY = int(os.environ.get("PHI_CONCURRENCY", "1"))
_phi_semaphore: asyncio.Semaphore | None = None


def _get_model():
    global _model, _tokenizer
    if _model is None:
        with _load_lock:
            if _model is None:  # re-check inside the lock
                logger.info(f"[model_file] Loading '{MODEL_ID}' ...")
                t0 = time.perf_counter()
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
                _model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    torch_dtype="auto",
                    device_map="auto",
                )
                _model.eval()
                logger.info(
                    f"[model_file] Model ready in {time.perf_counter() - t0:.2f}s "
                    f"(device={_model.device})"
                )
    return _model, _tokenizer


def generate_phi_response(messages: list[dict], max_new_tokens: int = 150) -> tuple[str, int, int, float]:
    """
    Blocking inference call. Loads the model on first use.
    Returns (response_text, input_tokens, output_tokens, elapsed_seconds).
    """
    model, tokenizer = _get_model()
    t0 = time.perf_counter()

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).to(model.device)
    input_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_tokens:]
    text          = tokenizer.decode(generated_ids, skip_special_tokens=True)
    elapsed       = time.perf_counter() - t0
    return text, input_tokens, len(generated_ids), elapsed


def _get_phi_semaphore() -> asyncio.Semaphore:
    global _phi_semaphore
    if _phi_semaphore is None:
        _phi_semaphore = asyncio.Semaphore(_PHI_CONCURRENCY)
    return _phi_semaphore


async def generate_phi_response_async(
    messages: list[dict], max_new_tokens: int = 150
) -> tuple[str, int, int, float]:
    """
    Async wrapper — runs the blocking generate() call in a thread pool so the
    event loop stays free, serialised through a semaphore so concurrent
    requests don't fight over the same model instance.
    """
    loop = asyncio.get_running_loop()
    async with _get_phi_semaphore():
        return await loop.run_in_executor(
            None, generate_phi_response, messages, max_new_tokens
        )
