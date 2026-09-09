from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "DocuLens AI"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: Literal["development", "testing", "production"] = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Storage & Upload Settings
    UPLOAD_DIR: str = "data/uploads"
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

    # Chunking Defaults
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50

    # Embedding Settings
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32

    # Visual Embedding Settings
    VISUAL_EMBEDDING_MODEL_NAME: str = "Qdrant/clip-ViT-B-32-vision"
    VISUAL_EMBEDDING_DIMENSION: int = 512
    VISUAL_EMBEDDING_BATCH_SIZE: int = 16


    # Vector Store (Qdrant) Settings
    QDRANT_LOCATION: str | None = ":memory:"
    QDRANT_PATH: str | None = None
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "document_chunks"
    QDRANT_VISUAL_COLLECTION_NAME: str = "visual_pages"


    # Retrieval Defaults
    DEFAULT_RETRIEVAL_TOP_K: int = 5
    DEFAULT_SCORE_THRESHOLD: float | None = None

    # Page Rendering Settings
    RENDER_DPI: int = 150
    RENDER_FORMAT: Literal["png", "jpeg"] = "png"
    RENDER_OUTPUT_DIR: str = "data/rendered_pages"

    # LLM & Generation Settings
    LLM_PROVIDER: Literal["mock", "gemini", "openai"] = "mock"
    LLM_MODEL: str = "gemini-3.7-flash"
    LLM_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    LLM_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1000

    # Optional secret placeholder (not hardcoded)
    SECRET_KEY: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


