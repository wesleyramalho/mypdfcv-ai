from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # OpenRouter — OpenAI-compatible gateway
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    agent_model: str = "google/gemini-2.0-flash-exp:free"
    judge_model: str = "google/gemini-2.0-flash-exp:free"

    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # DB
    database_url: str = f"sqlite:///{DATA_DIR}/mypdfcv_ai.db"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Logging
    log_level: str = "INFO"

    # Agent loop bounds
    agent_max_iterations: int = 12
    agent_temperature: float = 0.2


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
