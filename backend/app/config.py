import os
from pathlib import Path
from typing import Any
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "The Grand Azure Heritage Resort & Spa (Goa, India)"
    DEBUG: bool = True

    # LiteLLM Universal Model Selection
    # Examples:
    # - "gpt-4o-mini" (OpenAI)
    # - "gemini/gemini-2.0-flash" or "gemini/gemini-1.5-flash" (Google Gemini)
    # - "claude-3-5-sonnet-20241022" (Anthropic)
    # - "groq/llama-3.3-70b-versatile" (Groq)
    # - "ollama/llama3" (Local Ollama, zero key required)
    LLM_MODEL: str = "gpt-4o-mini"

    # API Keys for different providers
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    MOCK_LLM: bool = False
    HOTEL_DATA_PATH: Path = BASE_DIR / "app" / "data" / "hotel_data.json"
    CORS_ORIGINS: list[str] | str = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ]

    @field_validator("CORS_ORIGINS", mode="after")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

# Pass API keys into os.environ for LiteLLM
if settings.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
if settings.GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
if settings.ANTHROPIC_API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = settings.ANTHROPIC_API_KEY
if settings.GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY

# Determine if any key or local model is active
has_any_key = bool(
    settings.OPENAI_API_KEY.strip()
    or settings.GEMINI_API_KEY.strip()
    or settings.ANTHROPIC_API_KEY.strip()
    or settings.GROQ_API_KEY.strip()
    or settings.LLM_MODEL.startswith("ollama/")
)

if not has_any_key:
    settings.MOCK_LLM = True
