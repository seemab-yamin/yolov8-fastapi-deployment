from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Model
    model_path: str = "models/yolov8s.pt"
    device: str = "cuda"
    confidence_threshold: float = 0.25


settings = Settings()
