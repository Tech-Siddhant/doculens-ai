import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr

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
    assert settings.LOG_LEVEL == "INFO"
    assert settings.CORS_ORIGINS == "http://localhost:3000,http://127.0.0.1:3000"
    assert settings.RATE_LIMIT_ENABLED is False
    assert settings.RATE_LIMIT_MAX_REQUESTS == 60


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
        "LOG_LEVEL": "DEBUG",
        "CORS_ORIGINS": "http://localhost:3000,https://app.example.com",
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
        assert settings.LOG_LEVEL == "DEBUG"
        assert settings.CORS_ORIGINS == "http://localhost:3000,https://app.example.com"


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
        assert isinstance(custom_settings.GEMINI_API_KEY, SecretStr)
        assert custom_settings.GEMINI_API_KEY.get_secret_value() == "test-gemini-key"
        assert custom_settings.GEMINI_BASE_URL == "https://custom.gemini.endpoint/v1"


def test_secret_fields_are_secret_str() -> None:
    """API keys and credentials must be SecretStr so they aren't leaked in repr/logs."""
    env_vars = {
        "GEMINI_API_KEY": "sk-test-gemini",
        "LLM_API_KEY": "sk-test-llm",
        "QDRANT_API_KEY": "qdrant-secret",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        s = Settings(_env_file=None)
        for field_name in ("GEMINI_API_KEY", "LLM_API_KEY", "QDRANT_API_KEY"):
            val = getattr(s, field_name)
            assert isinstance(val, SecretStr), f"{field_name} should be SecretStr"
            # repr must NOT reveal the raw value
            assert env_vars[field_name] not in repr(val)
        # get_secret_value returns the raw string
        assert s.GEMINI_API_KEY.get_secret_value() == "sk-test-gemini"


def test_provider_key_required_gemini() -> None:
    """LLM_PROVIDER=gemini without a key must raise at construction time."""
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        Settings(_env_file=None, LLM_PROVIDER="gemini", GEMINI_API_KEY=None, LLM_API_KEY=None)


def test_provider_key_required_openai() -> None:
    """LLM_PROVIDER=openai without a key must raise at construction time."""
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        Settings(_env_file=None, LLM_PROVIDER="openai", LLM_API_KEY=None)


def test_mock_provider_needs_no_key() -> None:
    """LLM_PROVIDER=mock (default) succeeds without any API key."""
    settings = Settings(_env_file=None)
    assert settings.LLM_PROVIDER == "mock"
    assert settings.GEMINI_API_KEY is None
    assert settings.LLM_API_KEY is None


def test_invalid_environment_rejected() -> None:
    """ENVIRONMENT accepts only development/testing/production."""
    with patch.dict(os.environ, {"ENVIRONMENT": "staging"}, clear=False):
        with pytest.raises(Exception):
            Settings(_env_file=None)


