import os
from fastapi import Header, HTTPException, Depends
from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.environ.get("API_KEY", "")


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """FastAPI dependency — reads X-API-Key header and validates it."""
    if not _API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not configured on the server.")
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")
