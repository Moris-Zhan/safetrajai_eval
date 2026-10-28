"""
API keys for external services, read from the environment first and then from the
repo-root `.env` (git-ignored). Key values never appear in repr, str or error messages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from safetrajai.paths import PROJECT_ROOT

ENV_PATH = PROJECT_ROOT / ".env"

KEY_KINDS: Mapping[str, str] = {
    "GOOGLE_MAPS_DEMO_API_KEY": "demo",
    "GOOGLE_MAPS_API_KEY": "billing",
}

# Demo keys are for testing and prototyping only (Maps Demo Project Terms), so they are
# never used to produce base maps that analyses depend on.
GOOGLE_MAPS_KEYS_BY_PURPOSE: Mapping[str, Sequence[str]] = {
    "verify": ("GOOGLE_MAPS_DEMO_API_KEY", "GOOGLE_MAPS_API_KEY"),
    "geocode": ("GOOGLE_MAPS_DEMO_API_KEY", "GOOGLE_MAPS_API_KEY"),
    "basemap": ("GOOGLE_MAPS_API_KEY",),
}


class MissingApiKeyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ApiKey:
    name: str
    kind: str
    value: str = field(repr=False)

    def __str__(self) -> str:
        return f"{self.name} ({self.kind})"


def read_env_file(path: Path) -> Dict[str, str]:
    """Parse KEY=VALUE lines; comments, blank lines and empty values are skipped."""
    path = Path(path)
    if not path.exists():
        return {}
    entries: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            entries[name.strip()] = value
    return entries


def get_api_key(
    names: Sequence[str],
    env_path: Path = ENV_PATH,
    environ: Optional[Mapping[str, str]] = None,
) -> ApiKey:
    """First key found among `names`, checking the environment before the .env file."""
    environ = os.environ if environ is None else environ
    from_file = read_env_file(env_path)
    for name in names:
        value = environ.get(name) or from_file.get(name)
        if value:
            return ApiKey(name=name, kind=KEY_KINDS.get(name, "unknown"), value=value)
    raise MissingApiKeyError(f"找不到金鑰：請在環境變數或 {env_path} 設定 {' 或 '.join(names)}")


def google_maps_key(
    purpose: str,
    env_path: Path = ENV_PATH,
    environ: Optional[Mapping[str, str]] = None,
) -> ApiKey:
    if purpose not in GOOGLE_MAPS_KEYS_BY_PURPOSE:
        raise ValueError(f"未知的用途 {purpose!r}，可用：{sorted(GOOGLE_MAPS_KEYS_BY_PURPOSE)}")
    return get_api_key(GOOGLE_MAPS_KEYS_BY_PURPOSE[purpose], env_path=env_path, environ=environ)
