# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# # Load GPU model (Mistral 7B)
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True

# def generate_analysis(text):
#     """Generate advanced, professional 4-section PDF analysis using LLM"""

    


#     prompt = f"""
#     You are a professional document analyst capable of reviewing contracts, reports, policies, financial documents, academic papers, and general business PDFs.

#     Analyze the document and return ONLY valid JSON.

#     STRICT RULES:
#     - Do NOT explain anything.
#     - Do NOT add extra text.
#     - Output must be valid JSON only.

#     Return EXACTLY this format:

#     {{
#     "document_type": "Type of document (e.g., Contract, Report, Invoice, Policy, Research Paper, Other)",
#     "summary": "Concise executive summary",
#     "key_highlights": [
#     "Important point 1",
#     "Important point 2",
#     "Important point 3",
#     "Important point 4"
#     ],
#     "risk_analysis": [
#     "Risk or concern 1 (if applicable)",
#     "Risk or concern 2",
#     "Risk or concern 3",
#     "Risk or concern 4"
#     ]
#     }}

#     Document Content:
#     ----------------
#     {text}
#     ----------------

#     JSON:
#     """


#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     output = model.generate(
#         **inputs,
#         max_new_tokens=500,   # increased for detailed JSON
#         temperature=0.3,       # low for deterministic output
#         do_sample=False
#     )

# #    result = tokenizer.decode(output[0], skip_special_tokens=True)
#     result = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
#     return result


# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True


# def generate_analysis(text: str) -> str:

#     prompt = f"""<s>[INST]
# You are an expert document analyst with deep knowledge across legal, medical,
# financial, technical, academic, and business domains.

# Your task is to read any type of document and produce a structured JSON briefing
# that is accurate, professional, and immediately useful to the reader.

# EXAMPLE OUTPUT (follow this structure exactly):
# {{
#   "overview": "This is a [document type] about [main subject], intended for [audience/purpose].",
#   "summary": "4-6 sentence executive summary covering the document purpose, key content,
#                important findings or obligations, and conclusions or outcomes.",
#   "highlights": [
#     "Specific important fact, figure, date, obligation, finding, or condition from the document.",
#     "Specific important fact, figure, date, obligation, finding, or condition from the document.",
#     "Specific important fact, figure, date, obligation, finding, or condition from the document.",
#     "Specific important fact, figure, date, obligation, finding, or condition from the document."
#   ]
# }}

# FIELD RULES:

# "overview" — 1-2 sentences only.
#   - Identify what TYPE of document this is.
#     Examples of types: employment contract, research paper, medical report,
#     user manual, financial statement, privacy policy, invoice, academic thesis,
#     insurance policy, government notice, product specification, meeting minutes.
#   - State the subject, purpose, and intended audience or parties.

# "summary" — 4-6 sentences in professional, neutral tone.
#   - For CONTRACTS/LEGAL: cover parties, obligations, terms, penalties, duration.
#   - For MEDICAL/HEALTH: cover diagnosis, findings, recommendations, medications, follow-up.
#   - For FINANCIAL: cover revenue, expenses, profit/loss, forecasts, risks.
#   - For TECHNICAL/MANUAL: cover product purpose, key features, requirements, warnings.
#   - For RESEARCH/ACADEMIC: cover objective, methodology, key findings, conclusions.
#   - For REPORTS/BUSINESS: cover context, analysis, recommendations, outcomes.
#   - If document type is unclear, summarize the most important information a reader needs.

# "highlights" — 4 to 6 items.
#   - Each must be ONE complete, specific, factual sentence.
#   - Always include real values from the document: numbers, dates, names,
#     percentages, durations, prices, dosages, deadlines, versions, scores, or clauses.
#   - Do NOT write vague statements like "The document contains important information."
#   - Do NOT write opinions or analysis — only facts extracted directly from the document.
#   - Prioritize: critical obligations, risks, key figures, deadlines, warnings, or outcomes.

# STRICT OUTPUT RULES:
#   - Output ONLY the JSON object.
#   - Do NOT add any explanation, greeting, or commentary.
#   - Do NOT use markdown, code blocks, or backticks.
#   - Do NOT repeat these instructions in your output.
#   - All string values must be properly escaped.
#   - If a section has insufficient information, write "Not enough information available."

# Document:
# ----------------
# {text}
# ----------------
# [/INST]
# """

#     # Tokenize
#     encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
#     token_count = encoded["input_ids"].shape[1]

#     # Warn if truncation will occur
#     if token_count > 6000:
#         print(f"Warning: input is {token_count} tokens, truncating to 6000.")

#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         truncation=True,
#         max_length=6000
#     ).to("cuda")

#     # Generate
#     try:
#         output = model.generate(
#             **inputs,
#             max_new_tokens=600,
#             do_sample=False,
#             repetition_penalty=1.15,
#             pad_token_id=tokenizer.eos_token_id
#         )
#     except torch.cuda.OutOfMemoryError:
#         torch.cuda.empty_cache()
#         raise RuntimeError("GPU out of memory — reduce chunk size or document length.")

#     # Decode only new tokens
#     result = tokenizer.decode(
#         output[0][inputs["input_ids"].shape[1]:],
#         skip_special_tokens=True
#     )

#     return result




# updated code 

import torch
import asyncio
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Use CUDA if available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"


def _run_inference(prompt: str) -> str:
    """Synchronous inference — runs in a thread pool to avoid blocking the event loop."""

    # Tokenize
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
    token_count = encoded["input_ids"].shape[1]

    # Warn if truncation will occur
    if token_count > 6000:
        logger.warning(f"Input is {token_count} tokens, truncating to 6000.")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=6000
    ).to(device)

    # Generate
    try:
        output = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError("GPU out of memory — reduce chunk size or document length.")
    finally:
        # Free input tensors from GPU memory
        del inputs
        if device == "cuda":
            torch.cuda.empty_cache()

    # Decode only new tokens
    result = tokenizer.decode(
        output[0][encoded["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return result


async def generate_analysis(text: str) -> str:

    prompt = f"""<s>[INST]
You are an expert document analyst with deep knowledge across legal, medical,
financial, technical, academic, and business domains.

Your task is to read any type of document and produce a structured JSON briefing
that is accurate, professional, and immediately useful to the reader.

EXAMPLE OUTPUT (follow this structure exactly):
{{
  "overview": "This is a [document type] about [main subject], intended for [audience/purpose].",
  "summary": "4-6 sentence executive summary covering the document purpose, key content,
               important findings or obligations, and conclusions or outcomes.",
  "highlights": [
    "Specific important fact, figure, date, obligation, finding, or condition from the document.",
    "Specific important fact, figure, date, obligation, finding, or condition from the document.",
    "Specific important fact, figure, date, obligation, finding, or condition from the document.",
    "Specific important fact, figure, date, obligation, finding, or condition from the document."
  ]
}}

FIELD RULES:

"overview" — 1-2 sentences only.
  - Identify what TYPE of document this is.
    Examples of types: employment contract, research paper, medical report,
    user manual, financial statement, privacy policy, invoice, academic thesis,
    insurance policy, government notice, product specification, meeting minutes.
  - State the subject, purpose, and intended audience or parties.

"summary" — 4-6 sentences in professional, neutral tone.
  - For CONTRACTS/LEGAL: cover parties, obligations, terms, penalties, duration.
  - For MEDICAL/HEALTH: cover diagnosis, findings, recommendations, medications, follow-up.
  - For FINANCIAL: cover revenue, expenses, profit/loss, forecasts, risks.
  - For TECHNICAL/MANUAL: cover product purpose, key features, requirements, warnings.
  - For RESEARCH/ACADEMIC: cover objective, methodology, key findings, conclusions.
  - For REPORTS/BUSINESS: cover context, analysis, recommendations, outcomes.
  - If document type is unclear, summarize the most important information a reader needs.

"highlights" — 4 to 6 items.
  - Each must be ONE complete, specific, factual sentence.
  - Always include real values from the document: numbers, dates, names,
    percentages, durations, prices, dosages, deadlines, versions, scores, or clauses.
  - Do NOT write vague statements like "The document contains important information."
  - Do NOT write opinions or analysis — only facts extracted directly from the document.
  - Prioritize: critical obligations, risks, key figures, deadlines, warnings, or outcomes.

STRICT OUTPUT RULES:
  - Output ONLY the JSON object.
  - Do NOT add any explanation, greeting, or commentary.
  - Do NOT use markdown, code blocks, or backticks.
  - Do NOT repeat these instructions in your output.
  - All string values must be properly escaped.
  - If a section has insufficient information, write "Not enough information available."

Document:
----------------
{text}
----------------
[/INST]
"""

    # Run blocking inference in a thread pool to avoid blocking the async event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_inference, prompt)
    return result