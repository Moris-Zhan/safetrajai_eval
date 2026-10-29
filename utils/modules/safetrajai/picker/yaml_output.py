"""YAML text for the points clicked in the web picker (scripts/picker/make_picker.py).

Each function validates the points and returns text that the existing loaders read back
unchanged. Fields of config/homography/<camera>.yaml keep every top-level value on one line
so that update_camera_config_fields can still rewrite them later.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from safetrajai.collision.homography.camera_config import format_yaml_value

MIN_HOMOGRAPHY_POINTS = 4


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _point(value: Any, label: str) -> list:
    if not (isinstance(value, (list, tuple)) and len(value) == 2 and all(_number(c) for c in value)):
        raise ValueError(f"{label} 必須是 [x, y]（兩個數字），目前是 {value!r}")
    return [value[0], value[1]]


def _points(values: Any, label: str) -> list:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{label} 必須是點的清單，目前是 {values!r}")
    return [_point(value, f"{label}[{index}]") for index, value in enumerate(values)]


def _paired_points(first: Any, second: Any, first_label: str, second_label: str) -> tuple:
    first_points = _points(first, first_label)
    second_points = _points(second, second_label)
    if len(first_points) != len(second_points):
        raise ValueError(f"{first_label} 與 {second_label} 數量不符（{len(first_points)} 對 {len(second_points)}）")
    return first_points, second_points


def homography_yaml(camera: str, cctv_points: Sequence, map_points: Sequence) -> str:
    cctv, sat = _paired_points(cctv_points, map_points, "cctv_points", "map_points")
    if len(cctv) < MIN_HOMOGRAPHY_POINTS:
        raise ValueError(f"至少需要 {MIN_HOMOGRAPHY_POINTS} 組對應點（目前 {len(cctv)} 組）")
    return (
        f"# {camera}：Homography 對應點（CCTV 首幀像素 ↔ 衛星圖像素），貼進 config/homography/{camera}.yaml\n"
        f"cctv_points: {format_yaml_value(cctv)}\n"
        f"map_points: {format_yaml_value(sat)}\n"
    )


def checkpoints_yaml(camera: str, cctv_points: Sequence, sat_points: Sequence) -> str:
    cctv, sat = _paired_points(cctv_points, sat_points, "cctv_points", "sat_points")
    if not cctv:
        raise ValueError("至少需要 1 組驗證點")
    lines = [f"# {camera}：Homography 驗證點（check points），存成 config/homography_checkpoints/{camera}.yaml"]
    lines.append("cctv_points:")
    lines.extend(f"  - {format_yaml_value(point)}" for point in cctv)
    lines.append("sat_points:")
    lines.extend(f"  - {format_yaml_value(point)}" for point in sat)
    return "\n".join(lines) + "\n"


def camera_position_yaml(camera: str, view_point: Sequence) -> str:
    point = _point(view_point, "camera_view_point")
    return (
        f"# {camera}：相機在衛星圖上的位置（衛星圖像素），貼進 config/homography/{camera}.yaml\n"
        f"camera_view_point: {format_yaml_value(point)}\n"
    )


def _counting_line(line: Mapping[str, Any], index: int) -> dict:
    label = f"lines[{index}]"
    name = line.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{label}.name 不能是空的")
    points = _points(line.get("points"), f"{label}.points")
    if len(points) != 2:
        raise ValueError(f"{label}.points 必須剛好兩個點（目前 {len(points)} 個）")
    if points[0] == points[1]:
        raise ValueError(f"{label}.points 兩個點相同，線段長度為 0")
    result = {"name": name, "points": points}
    for key in ("forward_label", "backward_label"):
        if line.get(key) is not None:
            if not isinstance(line[key], str):
                raise ValueError(f"{label}.{key} 必須是文字")
            result[key] = line[key]
    return result


def counting_lines_yaml(camera: str, image_size: Sequence, lines: Sequence[Mapping[str, Any]]) -> str:
    if not (isinstance(image_size, (list, tuple)) and len(image_size) == 2
            and all(isinstance(s, int) and not isinstance(s, bool) and s > 0 for s in image_size)):
        raise ValueError(f"image_size 必須是兩個正整數 [寬, 高]，目前是 {image_size!r}")
    if not lines:
        raise ValueError("至少需要 1 條計數線")
    checked = [_counting_line(line, index) for index, line in enumerate(lines)]
    names = [line["name"] for line in checked]
    if len(set(names)) != len(names):
        raise ValueError(f"計數線名稱重複：{sorted({n for n in names if names.count(n) > 1})}")

    out = [
        f"# config/counting_lines/{camera}.yaml — 鏡頭 {camera} 的計數線（CCTV 首幀像素座標）",
        f"camera: {camera}",
        f"image_size: {format_yaml_value(list(image_size))}",
        "lines:",
    ]
    for line in checked:
        out.append(f"  - name: {format_yaml_value(line['name'])}")
        out.append(f"    points: {format_yaml_value(line['points'])}")
        for key in ("forward_label", "backward_label"):
            if key in line:
                out.append(f"    {key}: {format_yaml_value(line[key])}")
    return "\n".join(out) + "\n"
