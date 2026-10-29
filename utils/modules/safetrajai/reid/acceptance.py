"""追蹤驗收的設定覆寫、彙總與逐類別判定（spec: reid-model「只在追蹤不變差的類別啟用 Re-ID」；
change separability-gate-object-pairs D4）。

彙總方式：IDSW、CLR_TP 各序列加總；IDP＝ΣIDTP／(ΣIDTP＋ΣIDFP)；HOTA、AssA 依 GT_Dets 加權平均；
每幀追蹤時間依各序列幀數加權（由 stage1 log 的「Tracking complete — N frames」與「track avg : Xs」解析）。
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_FRAMES = re.compile(r"Tracking complete — (\d+) frames")
_TRACK_AVG = re.compile(r"track avg\s*:\s*([0-9.]+)s")


def parse_summary(path: Path) -> dict[str, float]:
    """TrackEval <class>_summary.txt: a header line of metric names and one line of values."""
    header, values = Path(path).read_text(encoding="utf-8").splitlines()[:2]
    return {name: float(value) for name, value in zip(header.split(), values.split())}


def aggregate(rows: list[dict]) -> dict:
    gt = sum(r["GT_Dets"] for r in rows)
    idtp = sum(r["IDTP"] for r in rows)
    idfp = sum(r["IDFP"] for r in rows)

    def weighted(key: str) -> float | None:
        return sum(r[key] * r["GT_Dets"] for r in rows) / gt if gt else None

    return {
        "IDSW": sum(r["IDSW"] for r in rows), "CLR_TP": sum(r["CLR_TP"] for r in rows), "GT_Dets": gt,
        "IDTP": idtp, "IDFP": idfp, "IDP": idtp / (idtp + idfp) if idtp + idfp else None,
        "HOTA": weighted("HOTA"), "AssA": weighted("AssA"),
    }


def judge_class(candidate: dict, baseline: dict) -> dict:
    """Enable only when nothing gets worse (IDSW, IDP, AssA, CLEAR TP) and identity improves (IDSW or IDP)."""
    if not baseline["GT_Dets"]:
        return {"passed": False, "reason": "調參序列沒有這個類別的 GT，無法判定"}
    worse = []
    if candidate["IDSW"] > baseline["IDSW"]:
        worse.append("IDSW 增加")
    if candidate["IDP"] < baseline["IDP"]:
        worse.append("IDP 下降")
    if candidate["AssA"] < baseline["AssA"]:
        worse.append("AssA 下降")
    if candidate["CLR_TP"] < baseline["CLR_TP"]:
        worse.append("CLEAR TP 下降")
    if worse:
        return {"passed": False, "reason": "、".join(worse)}
    if candidate["IDSW"] < baseline["IDSW"] or candidate["IDP"] > baseline["IDP"]:
        return {"passed": True, "reason": "不變差，且 IDSW 減少或 IDP 上升"}
    return {"passed": False, "reason": "沒有身分層級的改善（IDSW 未減少、IDP 未上升）"}


def parse_track_time(log_text: str) -> tuple[int, float | None]:
    """Frame-weighted mean of the per-sequence 'track avg' seconds in a stage1 log."""
    text = _ANSI.sub("", log_text)
    frames = [int(v) for v in _FRAMES.findall(text)]
    averages = [float(v) for v in _TRACK_AVG.findall(text)]
    total = sum(frames)
    if not total or len(frames) != len(averages):
        return 0, None
    return total, sum(f * a for f, a in zip(frames, averages)) / total


def apply_overrides(config: dict, sets: list[str]) -> dict:
    """Returns a copy of the scenario config with CLASS.KEY=VALUE overrides (VALUE parsed as YAML)."""
    result = copy.deepcopy(config)
    params = result["class_tracking"]["class_tracking_params"]
    for item in sets:
        target, sep, raw = item.partition("=")
        cls, dot, key = target.partition(".")
        if not sep or not dot or not cls or not key:
            raise ValueError(f"覆寫格式必須是 CLASS.KEY=VALUE：{item!r}")
        if cls not in params:
            raise ValueError(f"設定檔沒有類別 {cls!r}（{item!r}）")
        params[cls][key] = yaml.safe_load(raw)
    return result
