from pydantic import BaseModel
from typing import Optional

class TextTranslationRequest(BaseModel):
    text: str
    target_language: str
    source_language: Optional[str] = None  # Optional: auto-detect if not provided

class ChatRequest(BaseModel):
    message: str
    target_language: str