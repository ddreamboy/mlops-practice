from typing import Any

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RabbitMQSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RABBITMQ_")

    host: str = "localhost"
    port: int = 5672
    user: str = "guest"
    password: str = "guest"
    vhost: str = "/"

    @property
    def connection_url(self) -> str:
        return (
            f"amqp://{self.user}:{self.password}@{self.host}:{self.port}/{self.vhost}"
        )

    @property
    def backend_url(self) -> str:
        return f"rpc://{self.user}:{self.password}@{self.host}:{self.port}/{self.vhost}"


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = "localhost"
    port: int = 6379
    db: int = 0

    @property
    def connection_url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MODEL_")

    source: str = "local"
    path: str = "train/model.joblib"


class CelerySettings(BaseSettings):
    broker: str = ""
    backend: str = ""
    main: str = "emotion-classification"
    include: list[str] = ["src.tasks"]

    @model_validator(mode="after")
    def set_url_defaults(self) -> "CelerySettings":
        if not self.broker:
            self.broker = RabbitMQSettings().connection_url
        if not self.backend:
            self.backend = RedisSettings().connection_url
        return self

    @field_validator("include", mode="before")
    @classmethod
    def parse_include(cls, v: Any) -> list[str]:
        """
        Парсинг списка модулей из строки или списка.

        Args:
            v (Any): Значение поля include.

        Returns:
            list[str]: Список строк с именами модулей.
        """
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v


class GradioSettings(BaseSettings):
    api_base_url: str = "http://localhost:8000"
    poll_interval_seconds: float = 1.0
    poll_timeout_seconds: float = 30.0
