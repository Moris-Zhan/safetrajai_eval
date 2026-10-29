"""Parameters of the stage3 counting-line statistics, read from the scenario block `counting:`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Mapping


@dataclass(frozen=True)
class CountingConfig:
    enabled: bool = False


def load_counting_config(block: Optional[Any]) -> CountingConfig:
    """Validate the `counting:` block; a missing block means disabled."""
    if block is None:
        return CountingConfig()
    if not isinstance(block, Mapping):
        raise ValueError(f"counting 必須是對應表，目前是 {block!r}")
    unknown = sorted(set(block) - {"enabled"})
    if unknown:
        raise ValueError(f"counting 有未知欄位：{unknown}")
    enabled = block.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"counting.enabled 必須是 true 或 false，目前是 {enabled!r}")
    return CountingConfig(enabled=enabled)
