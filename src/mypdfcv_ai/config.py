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
    agent_model: str = "openai/gpt-oss-120b:free"
    # Comma-separated fallback chain. Tried in order on 429 / 503 from the
    # primary. Demonstrates the "graceful failure mechanisms" bullet on the JD.
    agent_fallback_models: str = (
        "qwen/qwen3-next-80b-a3b-instruct:free,"
        "meta-llama/llama-3.3-70b-instruct:free"
    )
    judge_model: str = "openai/gpt-oss-120b:free"

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
    agent_max_iterations: int = 8
    agent_temperature: float = 0.2

    # Per-LLM-call timeout in seconds. The OpenAI SDK default is 600s which
    # lets a single stuck free-tier call hang the whole agent loop; cap it
    # so the resilient client can fall back to the next model instead.
    llm_request_timeout_s: float = 30.0

    # FE integration. The /v1/tailor-resume endpoint is intended for
    # browser-originated traffic via the Next.js FE proxy, so:
    #   - tailor_api_token: shared secret the FE forwards in X-Tailor-Token.
    #     Empty string disables the check (local dev only).
    #   - allowed_origins: comma-separated CORS allowlist.
    tailor_api_token: str = ""
    allowed_origins: str = "http://localhost:3000"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
