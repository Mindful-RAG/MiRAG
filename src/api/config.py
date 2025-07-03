import os
from functools import lru_cache
from dotenv import load_dotenv

from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # Database settings
    DB_HOST: str = os.environ.get("DB_HOST", "")
    AWS_REGION: str = "ap-southeast-1"
    TABLE_NAME: str = os.environ.get("TABLE_NAME", "user-table")
    CHAT_STORE_TABLE_NAME: str = os.environ.get("CHAT_STORE_TABLE_NAME", "")
    AWS_ACCESS_KEY_ID: str = os.environ.get("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    PERSIST_PATH: str = os.environ.get("PERSIST_PATH", "./persisted_index")

    # Service URLs
    SEARXNG_URL: str = os.environ.get("SEARXNG_URL", "http://localhost:8003")
    FRONTEND_URL: str = os.environ.get("FRONTEND_URL", "http://localhost:3000")
    REDIRECT_URL: str = os.environ.get("REDIRECT_URL", "http://localhost:8000/auth")

    # Security settings
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "default_secret_key")
    JWT_SECRET_KEY: str = os.environ.get("JWT_SECRET_KEY", "very_secret_key")

    # Environment
    ENVIRONMENT: str = os.environ.get("ENVIRONMENT", "dev")

    # Storage
    BUCKET_NAME: str = os.environ.get("BUCKET_NAME", "")

    # Firebase settings
    GOOGLE_APPLICATION_CREDENTIALS: str = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    FIREBASE_PROJECT_ID: str = os.environ.get("FIREBASE_PROJECT_ID", "")
    FIREBASE_CLIENT_EMAIL: str = os.environ.get("FIREBASE_CLIENT_EMAIL", "")
    FIREBASE_PRIVATE_KEY: str = os.environ.get("FIREBASE_PRIVATE_KEY", "")

    # AI Model settings
    LLM_MODEL: str = "gpt-4o-mini"
    EMBED_MODEL: str = "BAAI/bge-large-en-v1.5"
    DATA_NAME: str = "nq"
    MAX_SIZE: int = 10 * 1024 * 1024  # 10 MB
    DEFAULT_CORPUS: str = "nq_corpus"

    # CORS settings
    # ALLOWED_ORIGINS: list = ["http://localhost:3000", "http://localhost:3001"]

    @property
    def ALLOWED_ORIGINS(self) -> list[str]:
        origins_str = os.environ.get("ALLOWED_ORIGINS", "")
        if origins_str:
            # Split by comma and strip whitespace
            origins = [origin.strip() for origin in origins_str.split(",") if origin.strip()]
            return origins

        # Default origins for development
        return ["http://localhost:3000", "http://localhost:3001"]

    # Authentication settings
    ACCESS_TOKEN_EXPIRE_HOURS: int = 24
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # Chat settings
    CHAT_TOKEN_LIMIT: int = 1500
    MAX_SESSIONS_PER_USER: int = 50


settings = Settings()


@lru_cache
def get_settings():
    return settings


env_vars = get_settings()
