"""
Class-specific tracking parameter configuration (loaded from YAML).

資料來源預設為與本模組同目錄的 ``track_config.yaml``。
執行 ``track_eval.py`` 或 ``scripts/grid_search/grid_search_*.py`` 時可用 ``--track-config`` 指定
其他 YAML；請在完整 argparse 之前呼叫 ``reload_track_config_from_cli()``，或於程式內
呼叫 ``load_track_config(path)``。

模組層級的 ``CLASS_NAMES``、``CLASS_TRACKING_PARAMS`` 等 dict 會以 clear/update 原地
更新，以便與 ``from .config import CLASS_TRACKING_PARAMS`` 的參考保持相容。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "需要 PyYAML 才能載入 track_config.yaml，請執行: pip install pyyaml"
    ) from e

from safetrajai.paths import CONFIG_DIR

# ---------------------------------------------------------------------------
# 由 load_track_config 填充（保留同一 dict 物件供其他模組 import 參考）
# ---------------------------------------------------------------------------
CLASS_NAMES: dict[int, str] = {}
CLASS_TRACKING_PARAMS: dict[str, dict[str, Any]] = {}
CLASS_CONF_THRESH: dict[str, float] = {}
DEFAULT_TRACKING_PARAMS: dict[str, Any] = {}
_DEFAULT_CONF_THRESH: list[float] = [0.15]  # 單元素 list 避免「整數重賦值」破壞 import 參考

_TRACK_CONFIG_RESOLVED: Path | None = None

DEFAULT_TRACK_CONFIG_PATH = CONFIG_DIR / "bytetrack.yaml"


def __getattr__(name: str) -> Any:
    """向後相容：``DEFAULT_CONF_THRESH`` 改為由 YAML 載入後的即時值。"""
    if name == "DEFAULT_CONF_THRESH":
        return _DEFAULT_CONF_THRESH[0]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_track_config_path() -> Path | None:
    """最近一次成功載入的 YAML 絕對路徑。"""
    return _TRACK_CONFIG_RESOLVED


def _resolve_yaml_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p.resolve()
    cwd = Path.cwd() / p
    if cwd.is_file():
        return cwd.resolve()
    beside = Path(__file__).resolve().parent / p.name
    if beside.is_file():
        return beside.resolve()
    pkg_rel = Path(__file__).resolve().parent / p
    if pkg_rel.is_file():
        return pkg_rel.resolve()
    raise FileNotFoundError(
        f"找不到 track config YAML: {path}（已嘗試絕對路徑、CWD、safetrajai/tracking/ 目錄）"
    )


def _coerce_class_names(raw: dict[Any, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for k, v in raw.items():
        out[int(k)] = str(v)
    return out


# Re-ID fusion parameters (spec: reid-fusion): name -> predicate on the numeric value, rule text
_FUSION_PARAM_RULES = {
    "reid_center_gate_px": (lambda v: v > 0, "> 0"),
    "reid_center_gate_px_per_frame": (lambda v: v >= 0, ">= 0"),
    "appearance_veto_sim": (lambda v: 0 <= v <= 1, "介於 0 與 1 之間"),
}

# Renamed fusion parameters (ADR 0008): old name -> new name
_RENAMED_FUSION_PARAMS = {"reid_center_gate_px_per_s": "reid_center_gate_px_per_frame"}


def _validate_fusion_params(class_name: str, params: dict[str, Any]) -> None:
    for old, new in _RENAMED_FUSION_PARAMS.items():
        if old in params:
            raise ValueError(f"class_tracking_params['{class_name}'].{old} 已不再支援，"
                             f"請改用 {new}（衍生量以保留期限幀數計算，見 ADR 0008）")
    for key, (is_valid, rule) in _FUSION_PARAM_RULES.items():
        if key not in params:
            continue
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not is_valid(value):
            raise ValueError(f"class_tracking_params['{class_name}'].{key} 必須是數字且 {rule}，目前為 {value!r}")


def _to_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} 必須是數字，目前為 {value!r}") from None


def load_track_config(path: str | Path | None = None) -> Path:
    """
    從 YAML 載入 tracking 設定，並寫入全域參數（class-aware tracking 用）。

    Parameters
    ----------
    path : str | Path | None
        YAML 檔案路徑
        - 若為 None → 使用 DEFAULT_TRACK_CONFIG_PATH（即根目錄 config.yaml 的 class_tracking 章節）

    Returns
    -------
    Path
        最終實際載入的 YAML 絕對路徑
    """

    # 宣告使用全域變數（記錄目前載入的 config 路徑）
    global _TRACK_CONFIG_RESOLVED

    # =========================
    # 1️⃣ 解析 YAML 路徑
    # =========================
    resolved = _resolve_yaml_path(path or DEFAULT_TRACK_CONFIG_PATH)

    # 讀取 YAML 檔
    with open(resolved, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # =========================
    # 2️⃣ 基本格式驗證（防呆）
    # =========================
    if not isinstance(data, dict):
        raise ValueError(f"track config 根節點必須為 mapping: {resolved}")

    # 若是 config/*.yaml 這種合併格式，實際內容在 class_tracking: 子章節；
    # 若是舊式獨立 track_config_*.yaml（扁平結構，已歸檔於 archive/track_config_variants/），
    # 沒有 class_tracking 鍵，直接用整份 data。
    data = data.get("class_tracking", data)

    # 取得各區塊
    cn = data.get("class_names")
    ctp = data.get("class_tracking_params")
    dtp = data.get("default_tracking_params")
    cct = data.get("class_conf_thresh")

    # 驗證 class_names
    if not isinstance(cn, dict) or not cn:
        raise ValueError(f"YAML 缺少非空 class_names: {resolved}")

    # 驗證 class_tracking_params
    if not isinstance(ctp, dict) or not ctp:
        raise ValueError(f"YAML 缺少非空 class_tracking_params: {resolved}")

    # 驗證 default_tracking_params
    if not isinstance(dtp, dict):
        raise ValueError(f"YAML 缺少 default_tracking_params mapping: {resolved}")

    # 驗證 class_conf_thresh
    if not isinstance(cct, dict) or not cct:
        raise ValueError(f"YAML 缺少非空 class_conf_thresh: {resolved}")

    # 驗證 Re-ID 融合參數（在更新任何全域參數之前）
    for name, params in ctp.items():
        if isinstance(params, dict):
            _validate_fusion_params(str(name), params)

    # Validate effective per-class retention before changing any loaded globals.
    for class_name in _coerce_class_names(cn).values():
        params = ctp.get(class_name, dtp)
        value = params.get("track_buffer") if isinstance(params, dict) else None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"類別 {class_name!r} 的 track_buffer 必須是正整數幀數，目前為 {value!r}")

    # 在修改任何全域參數之前，先完成所有驗證與型別轉換
    for name, params in ctp.items():
        # 每個 class 的設定必須是 dict（包含不在 class_names 的類別）
        if not isinstance(params, dict):
            raise ValueError(f"class_tracking_params['{name}'] 必須為 mapping")

    new_class_names = _coerce_class_names(cn)
    # 強制 key 為 string（避免 int / str 混亂）
    new_class_params = {str(name): dict(params) for name, params in ctp.items()}
    new_conf_thresh = {str(name): _to_float(val, f"class_conf_thresh['{name}']") for name, val in cct.items()}

    # 預設 conf threshold（若沒寫就用 0.15）
    dct = _to_float(data.get("default_conf_thresh", 0.15), "default_conf_thresh")

    # =========================
    # 3️⃣ 更新 CLASS_NAMES
    # =========================
    CLASS_NAMES.clear()
    CLASS_NAMES.update(new_class_names)

    # =========================
    # 4️⃣ 更新 CLASS_TRACKING_PARAMS
    # =========================
    CLASS_TRACKING_PARAMS.clear()
    CLASS_TRACKING_PARAMS.update(new_class_params)

    # =========================
    # 5️⃣ 更新 DEFAULT_TRACKING_PARAMS
    # =========================
    DEFAULT_TRACKING_PARAMS.clear()
    DEFAULT_TRACKING_PARAMS.update(dtp)

    # =========================
    # 6️⃣ 更新 CLASS_CONF_THRESH
    # =========================
    CLASS_CONF_THRESH.clear()
    CLASS_CONF_THRESH.update(new_conf_thresh)

    # =========================
    # 7️⃣ 設定 default conf threshold
    # =========================
    # ⚠️ 用 list 存是為了「可變引用」（避免 global immutable 問題）
    _DEFAULT_CONF_THRESH[0] = dct

    # =========================
    # 8️⃣ 記錄目前載入的 config
    # =========================
    _TRACK_CONFIG_RESOLVED = resolved

    # 回傳最終路徑
    return resolved


def reload_track_config_from_cli(argv: list[str] | None = None) -> Path:
    """
    僅解析 ``--track-config`` 後載入 YAML，供 ``if __name__ == '__main__'`` 在
    完整 ArgumentParser 之前呼叫；``argv`` 不含程式名稱（預設 ``sys.argv[1:]``）。
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--track-config",
        default=str(DEFAULT_TRACK_CONFIG_PATH),
        help=argparse.SUPPRESS,
    )
    ns, _ = p.parse_known_args(argv if argv is not None else sys.argv[1:])
    return load_track_config(ns.track_config)


def get_params_for_class(cls_id):
    """Get tracking parameters for a specific class ID."""
    if isinstance(cls_id, str):
        cls_name = cls_id
    else:
        cls_name = CLASS_NAMES.get(int(cls_id))

    if cls_name and cls_name in CLASS_TRACKING_PARAMS:
        return CLASS_TRACKING_PARAMS[cls_name]
    return DEFAULT_TRACKING_PARAMS


def get_class_name(cls_id):
    """Get class name from class ID."""
    return CLASS_NAMES.get(int(cls_id), f"class_{cls_id}")


def get_conf_thresh_for_class(cls_id):
    """Get confidence threshold for a specific class ID."""
    if isinstance(cls_id, str):
        cls_name = cls_id
    else:
        cls_name = CLASS_NAMES.get(int(cls_id))

    if cls_name and cls_name in CLASS_CONF_THRESH:
        return CLASS_CONF_THRESH[cls_name]
    return _DEFAULT_CONF_THRESH[0]


def get_min_conf_thresh():
    """Minimum confidence across classes (YOLO predict conf, then per-class filter)."""
    return min(CLASS_CONF_THRESH.values())


# 匯入模組時載入預設 YAML（未指定 --track-config 時與 CLI 預設一致）
load_track_config(DEFAULT_TRACK_CONFIG_PATH)
