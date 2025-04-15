import os
from functools import lru_cache
from dotenv import load_dotenv

from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    DB_HOST: str = "http://localhost:8040"
    AWS_REGION: str = "ap-southeast-1"
    PERSIST_PATH: str = os.environ.get("PERSIST_PATH", "./persisted_index")
    SEARXNG_URL: str = os.environ.get("SEARXNG_URL", "http://localhost:8003")
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "default_secret_key")
    ENVIRONMENT: str = os.environ.get("ENVIRONMENT", "dev")
    FRONTEND_URL: str = os.environ.get("FRONTEND_URL", "http://localhost:3000")
    REDIRECT_URL: str = os.environ.get("REDIRECT_URL", "http://localhost:8000/auth")
    JWT_SECRET_KEY: str = os.environ.get("JWT_SECRET_KEY", "very_secret_key")

    LLM_MODEL: str = "gpt-4o-mini"
    EMBED_MODEL: str = "BAAI/bge-large-en-v1.5"
    DATA_NAME: str = "nq"
    # AWS_ACCESS_KEY_ID: str = os.environ.get("AWS_ACCESS_KEY_ID", "")
    # AWS_SECRET_ACCESS_KEY: str = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    # GOOGLE_CLIENT_ID: str = os.environ.get("GOOGLE_CLIENT_ID", "")
    # GOOGLE_CLIENT_SECRET: str = os.environ.get("GOOGLE_CLIENT_SECRET", "")

    # model_config = SettingsConfigDict(env_file=".env")


settings = Settings()


@lru_cache
def get_settings():
    return settings


env_vars = get_settings()
