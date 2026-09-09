import os
from unittest.mock import patch

from app.core.config import Settings


def test_default_settings() -> None:
    settings = Settings(_env_file=None)
    assert settings.PROJECT_NAME == "DocuLens AI"
    assert settings.VERSION == "0.1.0"
    assert settings.API_V1_STR == "/api/v1"
    assert settings.ENVIRONMENT == "development"
    assert settings.DEBUG is True
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8000
    assert settings.UPLOAD_DIR == "data/uploads"
    assert settings.MAX_UPLOAD_SIZE_BYTES == 10 * 1024 * 1024
    assert settings.DEFAULT_CHUNK_SIZE == 500
    assert settings.DEFAULT_CHUNK_OVERLAP == 50
    assert settings.SECRET_KEY is None


def test_env_override_settings() -> None:
    env_vars = {
        "PROJECT_NAME": "Custom DocuLens",
        "ENVIRONMENT": "production",
        "DEBUG": "false",
        "PORT": "9000",
        "UPLOAD_DIR": "data/custom_uploads",
        "MAX_UPLOAD_SIZE_BYTES": "5242880",
        "DEFAULT_CHUNK_SIZE": "400",
        "DEFAULT_CHUNK_OVERLAP": "40",
        "SECRET_KEY": "supersecretkey",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        settings = Settings(_env_file=None)
        assert settings.PROJECT_NAME == "Custom DocuLens"
        assert settings.ENVIRONMENT == "production"
        assert settings.DEBUG is False
        assert settings.PORT == 9000
        assert settings.UPLOAD_DIR == "data/custom_uploads"
        assert settings.MAX_UPLOAD_SIZE_BYTES == 5242880
        assert settings.DEFAULT_CHUNK_SIZE == 400
        assert settings.DEFAULT_CHUNK_OVERLAP == 40
        assert settings.SECRET_KEY == "supersecretkey"

def test_gemini_settings_defaults_and_overrides() -> None:
    settings = Settings(_env_file=None)
    assert settings.LLM_PROVIDER == "mock"
    assert settings.LLM_MODEL == "gemini-3.7-flash"
    assert settings.GEMINI_BASE_URL == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert settings.GEMINI_API_KEY is None

    gemini_env = {
        "LLM_PROVIDER": "gemini",
        "LLM_MODEL": "gemini-3.7-flash",
        "GEMINI_API_KEY": "test-gemini-key",
        "GEMINI_BASE_URL": "https://custom.gemini.endpoint/v1",
    }
    with patch.dict(os.environ, gemini_env, clear=False):
        custom_settings = Settings(_env_file=None)
        assert custom_settings.LLM_PROVIDER == "gemini"
        assert custom_settings.LLM_MODEL == "gemini-3.7-flash"
        assert custom_settings.GEMINI_API_KEY == "test-gemini-key"
        assert custom_settings.GEMINI_BASE_URL == "https://custom.gemini.endpoint/v1"


