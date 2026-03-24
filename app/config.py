from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str = ""
    model_name: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3
    max_tokens: int = 4096


@lru_cache
def get_settings() -> Settings:
    return Settings()
