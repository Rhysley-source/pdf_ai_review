from string import Template
from typing import Dict, List

class PromptTemplates:
    """
    Central registry for all document prompt templates
    """

    # SYSTEM PROMPT 
    SYSTEM = """
You are an expert AI document reviewer.
Analyze documents carefully and return structured JSON output.
"""

    # DOCUMENT TEMPLATES
    DOCUMENT_TEMPLATES: Dict[str, Template] = {

        "resume": Template("""
Review the following resume.

Check for:
- Skills
- Education
- Projects
- Work Experience
- Achievements
- Missing sections
- Formatting issues
- Overall quality

[DOCUMENT]
$document_text

Return output strictly in JSON format:

{
  "missing_fields": [],
  "issues": [],
  "suggestions": [],
  "summary": ""
}
"""),

        "legal": Template("""
Review the following legal document.

Check for:
- Start date and End date
- Parties involved
- Key clauses
- Obligations and responsibilities
- Special clauses
- Risks and liabilities
- Missing or inconsistent terms

[DOCUMENT]
$document_text

Return output strictly in JSON format:

{
  "key_clauses": [],
  "risks": [],
  "issues": [],
  "summary": ""
}
"""),

        "general": Template("""
Review the following document.

Check for:
- Structure and clarity
- Key information
- Missing data
- Inconsistencies

[DOCUMENT]
$document_text

Return output strictly in JSON format:

{
  "issues": [],
  "suggestions": [],
  "summary": ""
}
"""),

        "signing_validation": Template("""
You are an expert document validator specializing in pre-signing checks.
Analyze the following document for completeness before it is signed.

AI Instructions:
1. Detect placeholders like "______", "[SIGN HERE]", "[DATE]", "[NAME]", etc.
2. Detect missing or incomplete fields that should be filled before signing.
3. Detect references to attachments, exhibits, or annexures (e.g., "Exhibit A", "Annexure 1", "Appendix B").
4. Cross-check these references against the provided list of attachments: $attachments_list
5. If a referenced attachment is NOT in the provided list, mark it as missing.

[DOCUMENT]
$document_text

Return output strictly in JSON format:
{
"missing_signature_fields": boolean,
"missing_dates": boolean,
"blank_fields": ["string"],
"missing_attachments": ["string"],
"alerts": ["string"]
}
"""),

        "categorize": Template("""
Analyze the document and classify it into one category:

- Resume
- Legal
- Invoice
- Report
- Other

[DOCUMENT]
$document_text

Return output strictly in JSON format:

{
  "category": ""
}
""")
    }

    # GET TEMPLATE (SAFE)
    @classmethod
    def get_template(cls, doc_type: str) -> Template:
        """
        Get template by document type.
        Falls back to 'general' if not found.
        """
        if not doc_type:
            return cls.DOCUMENT_TEMPLATES["general"]

        return cls.DOCUMENT_TEMPLATES.get(
            doc_type.lower(),
            cls.DOCUMENT_TEMPLATES["general"]
        )

 