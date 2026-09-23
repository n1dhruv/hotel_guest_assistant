from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "The Grand Azure Heritage Resort & Spa (Goa, India)"
    DEBUG: bool = True
    OPENAI_API_KEY: str = ""
    MOCK_LLM: bool = False
    HOTEL_DATA_PATH: Path = BASE_DIR / "app" / "data" / "hotel_data.json"
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

# If no OPENAI_API_KEY is provided, automatically fall back to MOCK_LLM mode
if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.strip() == "":
    settings.MOCK_LLM = True
