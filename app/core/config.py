from functools import lru_cache
from typing import Any, Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "DocuLens AI"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: Literal["development", "testing", "production"] = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Logging & CORS
    # Restrictive default: only the local frontend origins. "*" with credentials
    # effectively allows any origin; set explicitly only for trusted LAN/local tools.
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Storage & Upload Settings
    UPLOAD_DIR: str = "data/uploads"
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

    # Lightweight rate limiting (in-process, per client IP).
    # Disabled by default (single-user app); enable via env when exposed publicly.
    # ponytail: fixed-window in-memory limiter, per uvicorn worker. If run with
    # multiple workers or at scale, swap for a shared store (Redis) rate limiter.
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_MAX_REQUESTS: int = 60
    RATE_LIMIT_WINDOW_SECONDS: float = 60.0

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
    QDRANT_API_KEY: SecretStr | None = None
    QDRANT_COLLECTION_NAME: str = "document_chunks"
    QDRANT_VISUAL_COLLECTION_NAME: str = "visual_pages"


    # Retrieval Defaults
    DEFAULT_RETRIEVAL_TOP_K: int = 5
    DEFAULT_SCORE_THRESHOLD: float | None = None

    # BM25 Sparse Retrieval Settings
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    BM25_EPSILON: float = 0.25

    # Hybrid Retrieval Fusion Settings
    DEFAULT_HYBRID_STRATEGY: Literal["weighted", "rrf"] = "weighted"
    DEFAULT_WEIGHT_DENSE: float = 0.5
    DEFAULT_WEIGHT_BM25: float = 0.3
    DEFAULT_WEIGHT_VISUAL: float = 0.2
    DEFAULT_RRF_K: int = 60

    # Reranking Settings
    RERANKER_MODEL_NAME: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    RERANKER_BATCH_SIZE: int = 16
    DEFAULT_RERANK_TOP_K: int = 5

    # Evidence Selection Settings
    DEFAULT_EVIDENCE_TOP_K: int = 5

    # Page Rendering Settings
    RENDER_DPI: int = 150
    RENDER_FORMAT: Literal["png", "jpeg"] = "png"
    RENDER_OUTPUT_DIR: str = "data/rendered_pages"

    # LLM & Generation Settings
    # Context Assembly Settings
    DEFAULT_MAX_CONTEXT_CHARS: int = 16000


    LLM_PROVIDER: Literal["mock", "gemini", "openai"] = "mock"
    LLM_MODEL: str = "gemini-3.7-flash"
    LLM_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    LLM_API_KEY: SecretStr | None = None
    GEMINI_API_KEY: SecretStr | None = None
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1000
    LLM_TIMEOUT: float = 30.0
    LLM_MAX_RETRIES: int = 2

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        env_list_delimiter=",",
    )

    @model_validator(mode="after")
    def validate_provider_key_required(self) -> "Settings":
        """Fail fast and clearly when the selected provider needs an API key."""
        if self.LLM_PROVIDER == "gemini" and not (self.GEMINI_API_KEY or self.LLM_API_KEY):
            raise ValueError(
                "LLM_PROVIDER='gemini' requires GEMINI_API_KEY (or LLM_API_KEY) "
                "to be set via environment variables."
            )
        if self.LLM_PROVIDER == "openai" and not self.LLM_API_KEY:
            raise ValueError(
                "LLM_PROVIDER='openai' requires LLM_API_KEY to be set "
                "via environment variables."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


