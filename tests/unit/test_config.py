from app.core.config import Settings


def test_default_settings():
    config = Settings()
    assert config.PROJECT_NAME == "DocuLens AI"
    assert config.API_V1_STR == "/api/v1"
    assert config.ENV == "development"
    assert config.DEBUG is True


def test_settings_custom_env(monkeypatch):
    monkeypatch.setenv("PROJECT_NAME", "Custom DocuLens")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("API_V1_STR", "/api/v2")
    monkeypatch.setenv("ENV", "production")

    config = Settings()
    assert config.PROJECT_NAME == "Custom DocuLens"
    assert config.DEBUG is False
    assert config.API_V1_STR == "/api/v2"
    assert config.ENV == "production"
