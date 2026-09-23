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
    OPENROUTER_API_KEY: str = ""

    # Qdrant Vector DB Configuration
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "hotel_knowledge_base"
    QDRANT_STORAGE_PATH: Path = BASE_DIR / "app" / "data" / "qdrant_storage"
    QDRANT_IN_MEMORY: bool = False

    # Embedding Configuration: NVIDIA Nemotron-3-Embed-1B via OpenRouter (2048 dim, no fallback)
    EMBEDDING_PROVIDER: str = "openrouter"
    EMBEDDING_MODEL: str = "openrouter/nvidia/nemotron-3-embed-1b:free"
    EMBEDDING_DIMENSION: int = 2048
    FORCE_REINDEX: bool = False

    # HyDE (Hypothetical Document Embeddings) Query Pipeline
    HYDE_ENABLED: bool = True
    HYDE_MODEL: str = ""  # If empty, defaults to LLM_MODEL
    HYDE_MAX_TOKENS: int = 80
    HYDE_TEMPERATURE: float = 0.0
    HYDE_TIMEOUT: float = 2.5

    # Hybrid Search Pipeline (Dense Qdrant + Sparse BM25 + Reciprocal Rank Fusion)
    HYBRID_SEARCH_ENABLED: bool = True
    HYBRID_RRF_K: int = 60
    HYBRID_TOP_K: int = 10
    HYBRID_DENSE_CANDIDATES: int = 10
    HYBRID_BM25_CANDIDATES: int = 10

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
if settings.OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = settings.OPENROUTER_API_KEY

# Determine if any key or local model is active
has_any_key = bool(
    settings.OPENAI_API_KEY.strip()
    or settings.GEMINI_API_KEY.strip()
    or settings.ANTHROPIC_API_KEY.strip()
    or settings.GROQ_API_KEY.strip()
    or settings.OPENROUTER_API_KEY.strip()
    or settings.LLM_MODEL.startswith("ollama/")
)

if not has_any_key:
    settings.MOCK_LLM = True
