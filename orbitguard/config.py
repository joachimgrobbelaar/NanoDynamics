"""Application configuration and environment settings for OrbitGuard AI."""

import os
from pathlib import Path

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        PROJECT_NAME: str = "OrbitGuard AI"
        VERSION: str = "1.0.0"
        DEBUG: bool = False

        # Server Network
        HOST: str = "0.0.0.0"
        PORT: int = int(os.getenv("PORT", "8045"))

        # Database Configuration
        DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./orbitguard.db")

        # Redis Partitioning
        REDIS_URL: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/3")

        # Space Data Feeds
        CELESTRAK_BASE_URL: str = "https://celestrak.org/NORAD/elements/gp.php"
        SPACETRACK_USER: str = os.getenv("SPACETRACK_USER", "")
        SPACETRACK_PASSWORD: str = os.getenv("SPACETRACK_PASSWORD", "")

        # Conjunction & Astrodynamics Thresholds
        CONJUNCTION_THRESHOLD_KM: float = 10.0
        CRITICAL_MISS_DISTANCE_KM: float = 2.0
        COLLISION_PROB_THRESHOLD: float = 1e-4
        SCREENING_WINDOW_HOURS: int = 72
        DEFAULT_STEP_SIZE_SECONDS: float = 60.0
        FINE_STEP_SIZE_SECONDS: float = 1.0

        # Multi-Channel Alerting Settings
        TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
        DEFAULT_TELEGRAM_CHAT_ID: str = os.getenv("DEFAULT_TELEGRAM_CHAT_ID", "6491601588")
        WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "orbitguard_dev_secret_key")

        # LLM Briefing Settings
        LLM_PROVIDER: str = "ollama"
        OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

        model_config = SettingsConfigDict(
            env_file=str(Path(__file__).parent.parent.parent / ".env"),
            env_file_encoding="utf-8",
            extra="ignore",
        )

except ImportError:
    from pydantic import BaseModel

    class Settings(BaseModel):
        PROJECT_NAME: str = "OrbitGuard AI"
        VERSION: str = "1.0.0"
        DEBUG: bool = False

        HOST: str = "0.0.0.0"
        PORT: int = int(os.getenv("PORT", "8045"))
        DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./orbitguard.db")
        REDIS_URL: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/3")

        CELESTRAK_BASE_URL: str = "https://celestrak.org/NORAD/elements/gp.php"
        SPACETRACK_USER: str = os.getenv("SPACETRACK_USER", "")
        SPACETRACK_PASSWORD: str = os.getenv("SPACETRACK_PASSWORD", "")

        CONJUNCTION_THRESHOLD_KM: float = 10.0
        CRITICAL_MISS_DISTANCE_KM: float = 2.0
        COLLISION_PROB_THRESHOLD: float = 1e-4
        SCREENING_WINDOW_HOURS: int = 72
        DEFAULT_STEP_SIZE_SECONDS: float = 60.0
        FINE_STEP_SIZE_SECONDS: float = 1.0

        TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
        DEFAULT_TELEGRAM_CHAT_ID: str = os.getenv("DEFAULT_TELEGRAM_CHAT_ID", "6491601588")
        WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "orbitguard_dev_secret_key")

        LLM_PROVIDER: str = "ollama"
        OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")


settings = Settings()
