"""Unit tests for reading API keys from the environment and .env without leaking them."""

from pathlib import Path

import pytest

from safetrajai.secrets import MissingApiKeyError, get_api_key, google_maps_key, read_env_file

FAKE_DEMO = "fake-demo-value-123"
FAKE_BILLING = "fake-billing-value-456"


def write_env(tmp_path: Path, text: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_env_file_parsing_skips_comments_and_empty_values(tmp_path: Path) -> None:
    env = write_env(tmp_path, f"# comment\nGOOGLE_MAPS_API_KEY=\n\nGOOGLE_MAPS_DEMO_API_KEY={FAKE_DEMO}\n")
    assert read_env_file(env) == {"GOOGLE_MAPS_DEMO_API_KEY": FAKE_DEMO}


def test_environment_overrides_env_file(tmp_path: Path) -> None:
    env = write_env(tmp_path, f"GOOGLE_MAPS_API_KEY={FAKE_BILLING}\n")
    key = get_api_key(("GOOGLE_MAPS_API_KEY",), env_path=env, environ={"GOOGLE_MAPS_API_KEY": "from-env"})
    assert key.value == "from-env"


def test_key_kind_follows_variable_name(tmp_path: Path) -> None:
    env = write_env(tmp_path, f"GOOGLE_MAPS_DEMO_API_KEY={FAKE_DEMO}\nGOOGLE_MAPS_API_KEY={FAKE_BILLING}\n")
    assert google_maps_key("verify", env_path=env, environ={}).kind == "demo"
    assert google_maps_key("basemap", env_path=env, environ={}).kind == "billing"


def test_basemap_refuses_demo_only(tmp_path: Path) -> None:
    env = write_env(tmp_path, f"GOOGLE_MAPS_DEMO_API_KEY={FAKE_DEMO}\n")
    with pytest.raises(MissingApiKeyError) as excinfo:
        google_maps_key("basemap", env_path=env, environ={})
    assert FAKE_DEMO not in str(excinfo.value)
    assert "GOOGLE_MAPS_API_KEY" in str(excinfo.value)


def test_repr_hides_value(tmp_path: Path) -> None:
    env = write_env(tmp_path, f"GOOGLE_MAPS_DEMO_API_KEY={FAKE_DEMO}\n")
    key = google_maps_key("verify", env_path=env, environ={})
    assert FAKE_DEMO not in repr(key) and FAKE_DEMO not in str(key)
    assert key.value == FAKE_DEMO


def test_missing_key_is_error(tmp_path: Path) -> None:
    with pytest.raises(MissingApiKeyError):
        google_maps_key("verify", env_path=tmp_path / "absent.env", environ={})
