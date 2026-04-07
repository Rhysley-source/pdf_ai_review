# t_utils.py
import json
import re
import logging

logger = logging.getLogger(__name__)

def extract_json_from_text(text: str) -> dict:
    """
    Extracts a JSON object from a string.
    It looks for the first '{' and the last '}' to bound the JSON content.
    """
    logger.debug("Attempting to extract JSON from text.")

    # Find the start and end of the JSON block
    start_index = text.find('{')
    end_index = text.rfind('}')

    if start_index == -1 or end_index == -1 or end_index < start_index:
        logger.warning("Could not find a JSON block in the text.")
        return {}

    json_str = text[start_index : end_index + 1]

    # Clean up the string
    json_str = json_str.strip()
    # It might be wrapped in markdown
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    json_str = json_str.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON: {e}")
        logger.debug(f"Problematic JSON string: {json_str}")
        # As a fallback, try to be more robust with escapes
        json_str_repaired = re.sub(r'\(?!["\/bfnrtu])', r'', json_str)
        try:
            return json.loads(json_str_repaired)
        except json.JSONDecodeError as e2:
            logger.error(f"Failed to decode JSON even after repair: {e2}")
            return {}
