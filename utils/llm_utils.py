"""
llm_utils.py

Utility wrappers around the core LLM runner.
Provides run_llm_with_tokens for features that need token usage tracking.
"""

import logging
from llm_model.ai_model import _run_inference_text_obligation

logger = logging.getLogger(__name__)


async def run_llm_with_tokens(
    text: str,
    system_prompt: str,
) -> tuple[str, int, int]:
    """
    Generic plain-text LLM runner that also returns token counts.

    Same as run_llm in ai_model.py but additionally returns
    input_tokens and output_tokens for logging purposes.

    Args:
        text:          The document text to analyze.
        system_prompt: Instructions for the AI model.

    Returns:
        tuple of (response_text, input_tokens, output_tokens)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Document:\n----------------\n{text}\n----------------"},
    ]

    logger.debug("[run_llm_with_tokens] Sending request to LLM...")

    content, input_tokens, output_tokens = await _run_inference_text_obligation(
        messages,
        "run_llm_with_tokens"
    )

    logger.debug(
        f"[run_llm_with_tokens] Done — "
        f"input_tokens={input_tokens} output_tokens={output_tokens}"
    )

    return content, input_tokens, output_tokens