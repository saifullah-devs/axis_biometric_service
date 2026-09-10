from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Axis Pharma Biometric Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # InsightFace configuration
    INSIGHTFACE_MODEL_NAME: str = "buffalo_l"
    DETECTION_SIZE: int = 640
    SIMILARITY_THRESHOLD: float = 0.68
    MAX_IMAGE_SIZE_MB: int = 10

    # Execution Provider
    EXECUTION_PROVIDER: str = "CPUExecutionProvider"

    ORDS_FACE_LOGIN_URL: str = (
        "http://92.204.189.99:8080/ords/api/App/face-login"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()