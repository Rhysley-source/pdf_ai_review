# document_generation/prompt_templates.py


class SimulatedPromptTemplate:
    def __init__(self, template: str, input_variables: list[str]):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs) -> str:
        formatted_template = self.template
        for var in self.input_variables:
            if var in kwargs:
                formatted_template = formatted_template.replace(f"{{{var}}}", str(kwargs[var]))
        return formatted_template


# ---------------------------------------------------------------------------
# Universal document generation prompt — works for ANY document type.
# The LLM infers the document type from the user request itself.
# ---------------------------------------------------------------------------

DOCUMENT_GENERATION_PROMPT = SimulatedPromptTemplate(
    template="""You are an expert document generator.

STEP 1 — Check if the user's request is asking to generate, create, or draft any kind of document
(e.g. invoice, contract, resume, report, certificate, letter, agreement, proposal, purchase order, form, etc.).

If the request is NOT related to document generation, return ONLY this exact HTML and nothing else:
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{ font-family: Arial, sans-serif; font-size: 12pt; color: #000; background: #fff; margin: 40px; }}
  h2 {{ font-size: 14pt; margin-bottom: 12px; }}
  p {{ margin: 6px 0; }}
  ul {{ margin: 10px 0 0 20px; }}
  li {{ margin: 4px 0; }}
</style>
</head>
<body>
  <h2>Invalid Query</h2>
  <p>Your query does not appear to be related to document generation.</p>
  <p>Please provide a request to create a specific document. For example:</p>
  <ul>
    <li>Generate an invoice</li>
    <li>Create an NDA contract</li>
    <li>Draft an offer letter</li>
    <li>Make a sales report</li>
    <li>Create a purchase order</li>
    <li>Generate a certificate of completion</li>
  </ul>
</body>
</html>

STEP 2 — If the request IS document-related, generate a complete, professional HTML document.

User Request: {user_request}

Instructions:
- Identify the document type from the user's description.
- Generate a complete, well-structured HTML document that matches the identified type.
- The HTML must include <html>, <head> (with embedded <style>), and <body> tags.
- Add contenteditable='true' to the main content container inside <body> so the user can edit it directly in the browser.
- Use professional formatting, appropriate sections, and placeholder values where specific details are not provided.
- Do NOT include markdown backticks, explanations, or any text outside the HTML structure.
- Return ONLY the complete HTML document starting with <html> and ending with </html>.

STRICT DESIGN RULES — follow exactly:
- Layout must be flat and simple. No 3D effects, no shadows (no box-shadow, no text-shadow, no drop-shadow).
- No gradients (no linear-gradient, no radial-gradient, no background-image).
- No rounded corners (border-radius must be 0 or not used).
- No card/panel wrappers with padding and background that create a floating box effect.
- Body background must be plain white (#ffffff). No colored backgrounds on sections.
- The document content must fill the full page width. No centered narrow containers with large side margins.
- Use only these CSS properties for layout: width, padding, margin, border, font, color, text-align, display, table properties.
- Font: Arial or sans-serif, size 11pt–12pt for body text.
- Use simple horizontal lines (<hr> or border-bottom) to separate sections instead of colored blocks.
- Tables must use width:100%, simple 1px solid border, no background colors on rows.
- The result must look like a clean printed document — not a webpage widget.""",
    input_variables=["user_request"]
)


# ---------------------------------------------------------------------------
# Intent check prompt — quickly classifies whether a query is document-related.
# ---------------------------------------------------------------------------

INTENT_CHECK_PROMPT = SimulatedPromptTemplate(
    template="""You are a strict query classifier.

Decide whether the user's query is asking to generate, create, or draft any kind of document
(e.g. invoice, contract, resume, report, certificate, letter, agreement, proposal, form, etc.).

Reply with ONLY one word — exactly "YES" or "NO". No explanation, no punctuation.

User query: {user_request}""",
    input_variables=["user_request"]
)


# ---------------------------------------------------------------------------
# Regeneration prompt — modifies an existing HTML document.
# ---------------------------------------------------------------------------

REGENERATE_PROMPT = SimulatedPromptTemplate(
    template="""You are an expert HTML editor.
You will be provided with an existing HTML document and a user's request for modification.

GOAL:
Update the existing HTML content based on the user's instructions while maintaining the exact same CSS styles, design, and layout.

STRICT RULES:
1. Return ONLY the complete, updated HTML document starting with <html> and ending with </html>.
2. Ensure the main content container inside <body> remains contenteditable='true'.
3. Do NOT include markdown backticks (```html), explanations, or notes.
4. Preserve all original <style> blocks and CSS logic.
5. Apply the changes requested by the user accurately.
6. Do NOT introduce box-shadow, text-shadow, gradients, border-radius, or colored background panels — keep the design flat and print-ready.

Existing HTML:
{existing_html}

User Modification Request:
{modification_query}
""",
    input_variables=["existing_html", "modification_query"]
)
