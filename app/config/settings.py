from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    execution_mode: Literal["real", "mock", "replay"] = "mock"
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"
    judge_model: str = "qwen2.5:7b"
    temperature: float = 0.0
    replay_fixture_dir: str = "fixtures/replay"
    warning_accuracy_delta: float = 0.03
    critical_accuracy_delta: float = 0.08
    slack_webhook_url: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
