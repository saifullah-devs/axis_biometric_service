from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Axis Pharma Biometric Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # InsightFace configuration
    INSIGHTFACE_MODEL_NAME: str = "buffalo_l"
    DETECTION_SIZE: int = 640
    SIMILARITY_THRESHOLD: float = 0.68  # Industry/bank standard for ArcFace
    MAX_IMAGE_SIZE_MB: int = 10

    # 'CPUExecutionProvider' or 'CUDAExecutionProvider'
    EXECUTION_PROVIDER: str = "CPUExecutionProvider"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()